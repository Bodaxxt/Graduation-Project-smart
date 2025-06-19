import streamlit as st
import pyttsx3
import threading
import queue
import time
import sys
import os
from tempfile import NamedTemporaryFile
import datetime
import webbrowser
# try:
#     import pyautogui
# except ImportError:
#     pyautogui = None
#     print("pyautogui is not available in this environment")
import speech_recognition as sr
import json
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import numpy as np
import psutil
import subprocess
import vlc  # Kept for original script logic, less used by Streamlit directly for music
import yt_dlp
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from audiorecorder import audiorecorder
from tempfile import NamedTemporaryFile
import tensorflow as tf

try:
    from googletrans import Translator as GoogleTransService
except ImportError:
    print(
        "googletrans not found, using google_trans_new as fallback. Consider 'pip install googletrans==4.0.0rc1'"
    )
    from google_trans_new import google_translator

    GoogleTransService = google_translator

from googlesearch import search
from bs4 import BeautifulSoup
from gtts import gTTS
from playsound import (
    playsound,
)  # For original script logic, server-side reminders
import tempfile
from deep_translator import GoogleTranslator
import sqlite3
import re
from audiorecorder import audiorecorder
from pydub import AudioSegment

# Set correct paths for ffmpeg and ffprobe
# IMPORTANT: Replace with your actual paths if different
AUDIO_FFMPEG_PATH = r"E:\anaconda\Library\bin\ffmpeg.exe"  # Correct path to ffmpeg.exe
AUDIO_FFPROBE_PATH = r"E:\anaconda\Library\bin\ffprobe.exe"  # Correct path to ffprobe.exe
if os.path.exists(AUDIO_FFMPEG_PATH) and os.path.exists(AUDIO_FFPROBE_PATH):
    AudioSegment.converter = AUDIO_FFMPEG_PATH
    AudioSegment.ffprobe = AUDIO_FFPROBE_PATH
    print(f"FFmpeg and FFprobe paths set correctly: {AUDIO_FFMPEG_PATH}, {AUDIO_FFPROBE_PATH}")
else:
    print(f"ERROR: FFmpeg or FFprobe not found at the specified paths. Voice input may fail.\n"
          f"       Please ensure these paths are correct and the files exist.\n"
          f"       Current paths: ffmpeg - {AUDIO_FFMPEG_PATH}, ffprobe - {AUDIO_FFPROBE_PATH}")

# --- Global Variables (Many will be managed by st.session_state in Streamlit context) ---
OPENWEATHERMAP_API_KEY = "661f67d8cc9df0c9e161f20f23eb60d7"  # Your provided key
player = None  # VLC player instance
opened_processes = {}  # For tracking server-side processes
main_recognizer = sr.Recognizer()  # Speech recognizer instance
try:
    google_translate_service_instance = GoogleTransService()  # Translation service instance
except Exception as e_trans_init:
    print(f"Warning: Default GoogleTransService() init failed: {e_trans_init}.")
    from google_trans_new import google_translator

    GoogleTransService = google_translator
script_dir = os.path.dirname(
    os.path.abspath(sys.argv[0] if hasattr(sys, "frozen") else __file__)
)  # Script directory
tts_engine = None  # pyttsx3 engine
tts_engine_lock = threading.Lock()  # Lock for pyttsx3 engine
language_map = {
    "english": "en",
    "french": "fr",
    "german": "de",
    "korean": "ko",
    "spanish": "es",
    "chinese": "zh-cn",
    "japanese": "ja",
    "arabic": "ar",
    "hindi": "hi",
    "russian": "ru",
    "portuguese": "pt",
    "italian": "it",
}
# --- Database Initialization (Reminders Only) ---
DATABASE_NAME = os.path.join(script_dir, "assistant_data.db")  # Database file path

# --- Streamlit settings ---
logo_path = os.path.join(script_dir, "logo.jpeg")
st.set_page_config(
    page_title="VAIA - Voice Assistant",
    layout="wide",
    page_icon=logo_path  # استخدم مسار الصورة هنا
) # This MUST be the first Streamlit command

# --- Helper Functions ---


from spellchecker import SpellChecker

spell = SpellChecker()

def preprocess_input(text):
    """معالجة النص المدخل لتحسين التعرف على النوايا."""
    text = text.lower()  # تحويل النص إلى حروف صغيرة
    words = text.split()
    corrected_words = [spell.correction(word) or word for word in words]  # تصحيح الكلمات
    text = " ".join(corrected_words)
    text = re.sub(r'[^\w\s]', '', text)  # إزالة علامات الترقيم
    text = re.sub(r'\s+', ' ', text).strip()  # إزالة المسافات الزائدة
    return text

def init_db():
    """Initializes the SQLite database for reminders."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME, timeout=10, check_same_thread=False)
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS reminders (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     description TEXT NOT NULL,
                     datetime TEXT NOT NULL,
                     repeat TEXT NOT NULL)"""
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
        if "st" in globals():  # Check if Streamlit is running
            st.error(f"Database initialization error: {e}")
            st.stop()
        else:
            sys.exit(1)
    finally:
        if conn:
            conn.close()  # Ensure connection is closed


def initialize_pyttsx3_engine():
    """Initializes the pyttsx3 text-to-speech engine (server-side)."""
    global tts_engine  # Access the global tts_engine variable
    if tts_engine is None:  # Only initialize if it's not already initialized
        with tts_engine_lock:  # Use a lock to prevent race conditions in multithreaded environments
            if (
                tts_engine is None
            ):  # Double-check within the lock
                try:
                    engine = pyttsx3.init("sapi5")  # Initialize pyttsx3 engine (Windows)
                    if not engine:
                        print("TTS engine pyttsx3.init() failed.")
                        tts_engine = None
                        return None
                    voices = engine.getProperty("voices")
                    if voices:
                        engine.setProperty(
                            "voice",
                            voices[1].id if len(voices) > 1 else voices[0].id,
                        )  # Use a different voice if available
                    else:
                        print("No TTS voices found.")
                    rate = engine.getProperty("rate")
                    engine.setProperty(
                        "rate", max(50, rate - 70)
                    )  # Adjust speaking rate
                    volume = engine.getProperty("volume")
                    engine.setProperty(
                        "volume", min(1.0, volume + 0.25)
                    )  # Adjust volume
                    tts_engine = engine  # Set the global tts_engine
                except Exception as e:
                    print(f"Error initializing pyttsx3 engine: {e}")
                    tts_engine = None  # Handle initialization errors
    return tts_engine  # Return the engine instance


def speak_server_pyttsx3(text, wait=False):
    """Speaks text using pyttsx3 on the server."""
    global tts_engine_lock, tts_engine
    engine_instance = initialize_pyttsx3_engine()  # Get engine instance
    if engine_instance:
        with tts_engine_lock:  # Use a lock to prevent race conditions
            try:
                engine_instance.say(text)
                engine_instance.runAndWait()
            except RuntimeError:  # Handle runtime errors
                print("TTS Runtime Error. Reinitializing engine.")
                tts_engine = None  # Reset global engine
                engine_instance_retry = initialize_pyttsx3_engine()  # Reinitialize engine
                if engine_instance_retry:
                    try:
                        engine_instance_retry.say(text)
                        engine_instance_retry.runAndWait()
                    except Exception as e_retry:
                        print(f"TTS Error on retry: {e_retry}")  # Handle retry errors
                else:
                    print("TTS reinitialization failed.")
            except Exception as e:
                print(f"Error during server speech: {e}")  # Handle general errors
        if wait:
            time.sleep(0.5)
    else:
        print(f"SERVER TTS SKIPPED (engine not initialized): {text}")


def generate_assistant_response_audio(
    text_to_speak, lang_code="en", force_speak_st=False
):
    """Generates audio using gTTS and returns a message dictionary for Streamlit."""
    assistant_responses = []  # To hold text and audio data for Streamlit UI

    if (
        st.session_state.get("is_music_playing_st", False) and not force_speak_st
    ):
        print(
            f"[Speech Suppressed in ST] Music is playing. Suppressed: '{text_to_speak}'"
        )
        assistant_responses.append(
            {
                "role": "assistant",
                "text_content": f"(Speech suppressed as music is playing): {text_to_speak}",
                "audio_path": None,
            }
        )  # Add a message indicating speech suppression
        return assistant_responses  # Return immediately

    audio_file_path = None
    try:
        tts = gTTS(
            text=text_to_speak, lang=lang_code, slow=False
        )  # Generate speech with gTTS
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp3", dir=tempfile.gettempdir()
        ) as fp:
            audio_file_path = (
                fp.name
            )  # Get temporary file path
        tts.save(audio_file_path)  # Save audio to temporary file
        print(
            f"Generated audio for Streamlit: '{text_to_speak}' in {lang_code} at {audio_file_path}"
        )
    except Exception as e:
        print(f"gTTS error for Streamlit: {e}")
        if force_speak_st:  # Only for critical forced messages, use server-side TTS
            speak_server_pyttsx3(
                f"Error generating speech for: {text_to_speak}. Error was: {e}"
            )
        assistant_responses.append(
            {
                "role": "assistant",
                "text_content": f"{text_to_speak} (Audio generation failed: {e})",
                "audio_path": None,
            }
        )  # Add a message indicating audio generation failure
        return assistant_responses

    assistant_responses.append(
        {
            "role": "assistant",
            "text_content": text_to_speak,
            "audio_path": audio_file_path,
        }
    )  # Create the assistant message
    return assistant_responses  # Return the list of responses


def speak_random_st(phrases, force_speak_st=False):
    """Selects a random phrase and generates audio for it."""
    return generate_assistant_response_audio(
        random.choice(phrases), force_speak_st=force_speak_st
    )


def command_py(recognizer_instance=None, prompt_text=None, force_prompt_for_st_speak=False):
    """Listens for audio input using the server's microphone and transcribes it."""
    generated_prompts = []  # List to store generated prompts
    current_recognizer = (
        recognizer_instance if recognizer_instance else main_recognizer
    )  # Use provided recognizer or default

    if prompt_text:
        print(f"Prompting (server-mic listen): {prompt_text}")

    try:
        with sr.Microphone() as source:  # Use the microphone as audio source
            try:
                current_recognizer.adjust_for_ambient_noise(
                    source, duration=0.5
                )  # Adjust for noise
            except Exception as e_adjust:
                print(f"Could not adjust for ambient noise: {e_adjust}")

            st.toast(
                "🎙️ Listening via server microphone...", icon="🎤"
            )  # Display listening message
            print("Listening (server mic)...", end="", flush=True)
            current_recognizer.pause_threshold = 0.8  # Set pause threshold
            audio = current_recognizer.listen(
                source, timeout=7, phrase_time_limit=12
            )  # Listen for audio

            st.toast("🤔 Recognizing speech...", icon="🤔")  # Display recognizing message
            print("\rRecognizing (server mic)...", end="", flush=True)
            query = current_recognizer.recognize_google(
                audio, language="en-US"
            )  # Recognize speech
            print(f"\rUser said (server mic): {query}\n")
            return query.lower(), generated_prompts  # Return the recognized query
    except sr.WaitTimeoutError:
        print("\rNo speech detected (server mic).")
        generated_prompts.extend(
            generate_assistant_response_audio(
                "I didn't hear anything from the server microphone.",
                force_speak_st=True,
            )
        )
        return None, generated_prompts
    except sr.UnknownValueError:
        print("\rCould not understand audio (server mic).")
        generated_prompts.extend(
            generate_assistant_response_audio(
                "Sorry, I couldn't understand what was said into the server microphone.",
                force_speak_st=True,
            )
        )
        return "unintelligible", generated_prompts
    except sr.RequestError as e:
        print(f"\rSpeech service request error (server mic); {e}")
        generated_prompts.extend(
            speak_random_st(
                [
                    "My speech recognition (server-side) is offline.",
                    "Connection issue with speech service (server-side).",
                ],
                force_speak_st=True,
            )
        )
        return None, generated_prompts
    except OSError as e_os:
        print(f"\rOSError with server microphone: {e_os}. Check microphone.")
        generated_prompts.extend(
            generate_assistant_response_audio(
                "Server microphone access error. Please check its connection.",
                force_speak_st=True,
            )
        )
        return None, generated_prompts
    except Exception as e:
        print(f"\rUnexpected error in command_py(): {e}")
        import traceback

        traceback.print_exc()
        generated_prompts.extend(
            generate_assistant_response_audio(
                "An unexpected error occurred with server voice input.",
                force_speak_st=True,
            )
        )
        return None, generated_prompts


# --- Database Functions (Reminders) ---
def add_reminder_db(desc, dt_string, repeat_type):
    """Adds a reminder to the database."""
    conn = None
    try:
        conn = sqlite3.connect(
            DATABASE_NAME, timeout=10, check_same_thread=False
        )  # Connect to database
        c = conn.cursor()
        c.execute(
            "INSERT INTO reminders (description, datetime, repeat) VALUES (?, ?, ?)",
            (desc, dt_string, repeat_type),
        )  # Insert reminder
        conn.commit()
    except sqlite3.Error as e:
        print(f"DB Error adding reminder: {e}")
    finally:
        if conn:
            conn.close()  # Ensure connection is closed


def remove_reminder_db(reminder_id):
    """Removes a reminder from the database."""
    conn = None
    deleted_count = 0
    try:
        conn = sqlite3.connect(
            DATABASE_NAME, timeout=10, check_same_thread=False
        )  # Connect to database
        c = conn.cursor()
        c.execute(
            "DELETE FROM reminders WHERE id = ?", (reminder_id,)
        )  # Delete reminder
        deleted_count = c.rowcount
        conn.commit()
    except sqlite3.Error as e:
        print(f"DB Error removing reminder: {e}")
    finally:
        if conn:
            conn.close()  # Ensure connection is closed
    return deleted_count > 0


def get_reminders_db():
    """Retrieves all reminders from the database."""
    conn = None
    reminders_list = []
    try:
        conn = sqlite3.connect(
            DATABASE_NAME, timeout=10, check_same_thread=False
        )  # Connect to database
        c = conn.cursor()
        c.execute(
            "SELECT id, description, datetime, repeat FROM reminders ORDER BY datetime"
        )  # Select reminders
        reminders_list = c.fetchall()
    except sqlite3.Error as e:
        print(f"DB Error getting reminders: {e}")
    finally:
        if conn:
            conn.close()  # Ensure connection is closed
    return reminders_list


def parse_month(text):
    """Parses a month name or number from text."""
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    text_lower = text.lower().strip()  # Normalize text
    for name, num in months.items():
        if name in text_lower:
            return num  # Check for month names
    match_num = re.search(
        r"\b([1-9]|1[0-2])\b", text_lower
    )  # Check for month numbers
    if match_num:
        return int(match_num.group(1))
    return None


# Reminder Management Flow for Streamlit
# This will be a state machine managed by st.session_state
def manage_reminders_flow_st(user_action_or_detail):
    """Manages the reminder creation/viewing/removal flow in Streamlit."""
    responses = []
    # st.session_state.reminder_flow_step: None, "action_prompted", "add_desc", "add_date", "add_time", "add_repeat", "view", "remove_prompt", "remove_confirm"
    # st.session_state.reminder_data: {} to store parts of reminder being built

    if "reminder_flow_step" not in st.session_state:  # Initialize flow state if not present
        st.session_state.reminder_flow_step = "action_prompted"
        st.session_state.reminder_data = {}  # Store reminder details
        responses.extend(
            generate_assistant_response_audio(
                "What would you like to do with reminders? You can add, view, or remove a reminder.",
                force_speak_st=True,
            )
        )
        return responses

    step = st.session_state.reminder_flow_step  # Current step in the flow
    action = (
        user_action_or_detail.lower() if user_action_or_detail else ""
    )  # User's action (or detail)

    if step == "action_prompted":
        if (
            not action or action == "unintelligible"
        ):  # If no action or unintelligible input
            responses.extend(
                generate_assistant_response_audio(
                    "I didn't catch that reminder action. Please say add, view, or remove.",
                    force_speak_st=True,
                )
            )
        elif "add" in action:
            st.session_state.reminder_flow_step = "add_desc"
            responses.extend(
                generate_assistant_response_audio(
                    "What is the reminder about?", force_speak_st=True
                )
            )
        elif "view" in action or "show" in action or "list" in action:
            st.session_state.reminder_flow_step = "view"
            reminders = get_reminders_db()  # Get reminders from DB
            if not reminders:  # If no reminders
                responses.extend(
                    generate_assistant_response_audio(
                        "You have no reminders set.", force_speak_st=True
                    )
                )
                st.session_state.reminder_flow_step = None  # End flow
            else:
                responses.extend(
                    generate_assistant_response_audio(
                        "Here are your reminders:", force_speak_st=True
                    )
                )
                for r_id, desc, dt_str, repeat in reminders:  # List reminders
                    try:
                        dt_obj = datetime.datetime.fromisoformat(
                            dt_str
                        )  # Parse datetime
                        responses.extend(
                            generate_assistant_response_audio(
                                f"Number {r_id}: {desc}, on {dt_obj.strftime('%A, %B %d at %I:%M %p')}. Repeats: {repeat}.",
                                force_speak_st=True,
                            )
                        )
                    except ValueError:
                        responses.extend(
                            generate_assistant_response_audio(
                                f"Number {r_id}: {desc}, at invalid time '{dt_str}'. Repeats: {repeat}.",
                                force_speak_st=True,
                            )
                        )
                st.session_state.reminder_flow_step = None  # End flow
        elif "remove" in action or "delete" in action:
            reminders = get_reminders_db()  # Get reminders from DB
            if not reminders:  # If no reminders
                responses.extend(
                    generate_assistant_response_audio(
                        "You have no reminders to remove.", force_speak_st=True
                    )
                )
                st.session_state.reminder_flow_step = None  # End flow
            else:
                st.session_state.reminder_flow_step = "remove_prompt"
                st.session_state.reminders_for_removal = (
                    reminders
                )  # Store reminders for removal
                responses.extend(
                    generate_assistant_response_audio(
                        "Here are your reminders. Which one to remove? Say the number or part of the description.",
                        force_speak_st=True,
                    )
                )
                for r_id, desc, dt_str, repeat in reminders:  # List reminders
                    try:
                        dt_obj = datetime.datetime.fromisoformat(
                            dt_str
                        )  # Parse datetime
                        responses.extend(
                            generate_assistant_response_audio(
                                f"Number {r_id}: {desc} ({dt_obj.strftime('%b %d, %I:%M%p')})",
                                force_speak_st=True,
                            )
                        )
                    except ValueError:
                        responses.extend(
                            generate_assistant_response_audio(
                                f"Number {r_id}: {desc} (invalid date)",
                                force_speak_st=True,
                            )
                        )
        else:
            responses.extend(
                generate_assistant_response_audio(
                    "Sorry, I didn't understand that reminder action. Try 'add', 'view', or 'remove'.",
                    force_speak_st=True,
                )
            )
            st.session_state.reminder_flow_step = None  # Reset flow

    elif step == "add_desc":
        if not action:  # If description is empty
            responses.extend(
                generate_assistant_response_audio(
                    "Description cannot be empty. Cancelling reminder addition.",
                    force_speak_st=True,
                )
            )
            st.session_state.reminder_flow_step = None
            st.session_state.reminder_data = {}  # Reset flow
        else:
            st.session_state.reminder_data[
                "desc"
            ] = action  # Store description
            st.session_state.reminder_flow_step = "add_date"
            responses.extend(
                generate_assistant_response_audio(
                    "What date? (e.g., 'tomorrow', 'next Friday', 'July 25th', or YYYY-MM-DD)",
                    force_speak_st=True,
                )
            )

    elif step == "add_date":
        parsed_date_obj = None
        if not action:  # If date is not provided
            responses.extend(
                generate_assistant_response_audio(
                    "Date not provided. Cancelling.", force_speak_st=True
                )
            )
            st.session_state.reminder_flow_step = None
            st.session_state.reminder_data = {}  # Reset flow
            return responses

        today = datetime.date.today()
        date_input_lower = action.lower()
        if "today" in date_input_lower:
            parsed_date_obj = today  # Today
        elif "tomorrow" in date_input_lower:
            parsed_date_obj = today + datetime.timedelta(
                days=1
            )  # Tomorrow
        elif "next" in date_input_lower:  # Next weekday
            days_of_week = [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ]
            try:
                target_day_name = next(
                    day for day in days_of_week if day in date_input_lower
                )
                days_ahead = days_of_week.index(target_day_name) - today.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                parsed_date_obj = today + datetime.timedelta(days=days_ahead)
            except StopIteration:
                pass
        if not parsed_date_obj:
            try:
                parsed_date_obj = datetime.datetime.strptime(
                    action, "%Y-%m-%d"
                ).date()  # YYYY-MM-DD format
            except ValueError:
                pass
        if not parsed_date_obj:
            try:  # Month Day Year format
                match_mdy = re.search(
                    r"(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?",
                    action,
                    re.IGNORECASE,
                )
                if match_mdy:
                    month_name, day_num_str, year_str = match_mdy.groups()
                    month_num = parse_month(month_name)
                    if month_num:
                        day_num = int(day_num_str)
                        year_num = int(year_str) if year_str else today.year
                        parsed_date_obj = datetime.date(
                            year_num, month_num, day_num
                        )
                        if not year_str and parsed_date_obj < today:
                            parsed_date_obj = datetime.date(
                                today.year + 1, month_num, day_num
                            )  # Assume next year
            except:
                pass

        if not parsed_date_obj:  # Date could not be parsed
            responses.extend(
                generate_assistant_response_audio(
                    "Could not understand date. Try 'Month Day Year' or 'YYYY-MM-DD', 'tomorrow', etc. Please provide the date again.",
                    force_speak_st=True,
                )
            )
        else:
            st.session_state.reminder_data["date_str"] = (
                parsed_date_obj.isoformat()
            )  # Store date in ISO format
            st.session_state.reminder_flow_step = "add_time"
            responses.extend(
                generate_assistant_response_audio(
                    "What time? (e.g., 9 AM, 2:30 PM, 14:00)",
                    force_speak_st=True,
                )
            )

    elif step == "add_time":
        time_obj = None
        if not action:  # If time not provided
            responses.extend(
                generate_assistant_response_audio(
                    "Time not provided. Cancelling.", force_speak_st=True
                )
            )
            st.session_state.reminder_flow_step = None
            st.session_state.reminder_data = {}  # Reset flow
            return responses

        time_input_raw = action
        time_input = time_input_raw.strip().upper()  # Normalize time input
        time_input = (
            time_input.replace("A.M.", "AM").replace("P.M.", "PM")
        )  # Standardize AM/PM
        if time_input == "12 AM" or time_input == "12AM":
            time_obj = datetime.time(0, 0)  # Midnight
        elif time_input == "12 PM" or time_input == "12PM":
            time_obj = datetime.time(12, 0)  # Noon
        elif "NOON" in time_input:
            time_obj = datetime.time(12, 0)  # Noon
        elif "MIDNIGHT" in time_input:
            time_obj = datetime.time(0, 0)  # Midnight

        if not time_obj:
            match_24h_with_ampm = re.match(
                r"(\d{1,2}):(\d{2})\s*(AM|PM)", time_input
            )  # 24h format with AM/PM
            if match_24h_with_ampm:
                hour_str, minute_str, ampm_str = match_24h_with_ampm.groups()
                hour_val = int(hour_str)
                if hour_val > 12:
                    time_input = f"{hour_str}:{minute_str}"

            time_formats = ["%I:%M %p", "%I%p", "%H:%M", "%H"]  # Time formats to try
            time_input_processed = time_input.replace(".", ":").replace(
                " ", ""
            )  # Normalize time input
            # Try to parse "9 PM" or "9AM etc.
            match_hour_ampm = re.match(
                r"(\d{1,2})\s*(AM|PM)", time_input_processed
            )
            if match_hour_ampm and ":" not in time_input_processed:  # e.g. 9PM, 10 AM
                hour_str, ampm_str = match_hour_ampm.groups()
                time_input_processed = f"{hour_str}:00{ampm_str}"

            for fmt in time_formats:  # Try parsing time
                try:
                    time_obj = datetime.datetime.strptime(
                        time_input_processed, fmt
                    ).time()
                    break
                except ValueError:
                    try:
                        time_obj = datetime.datetime.strptime(
                            time_input, fmt
                        ).time()  # Try with original input
                        break
                    except ValueError:
                        continue  # If parsing fails, try next format

        if not time_obj:  # If time could not be parsed
            responses.extend(
                generate_assistant_response_audio(
                    "Invalid time. Try HH:MM, H AM/PM (e.g. 9 AM, 2:30 PM, 14:00). Please provide the time again.",
                    force_speak_st=True,
                )
            )
        else:
            st.session_state.reminder_data["time_str_iso"] = time_obj.strftime(
                "%H:%M:%S"
            )  # Store time in ISO format
            st.session_state.reminder_flow_step = "add_repeat"
            responses.extend(
                generate_assistant_response_audio(
                    "Repeat? (once, daily, weekly, or none)", force_speak_st=True
                )
            )

    elif step == "add_repeat":
        repeat_type = "once"  # Default repeat type
        if action:
            repeat_input_lower = action.lower()
            if "daily" in repeat_input_lower:
                repeat_type = "daily"  # Daily repeat
            elif "weekly" in repeat_input_lower:
                repeat_type = "weekly"  # Weekly repeat
        st.session_state.reminder_data["repeat_type"] = repeat_type  # Store repeat type

        try:
            final_dt_obj = datetime.datetime.fromisoformat(
                f"{st.session_state.reminder_data['date_str']}T{st.session_state.reminder_data['time_str_iso']}"
            )  # Combine date and time
            if (
                final_dt_obj <= datetime.datetime.now() + datetime.timedelta(minutes=1)
            ):  # Check if reminder is in the past or too soon
                responses.extend(
                    generate_assistant_response_audio(
                        "The reminder time is in the past or too soon. Please start over.",
                        force_speak_st=True,
                    )
                )
            else:
                add_reminder_db(
                    st.session_state.reminder_data["desc"],
                    final_dt_obj.isoformat(),
                    st.session_state.reminder_data["repeat_type"],
                )  # Add reminder to database
                responses.extend(
                    generate_assistant_response_audio(
                        f"Okay, I've set a {st.session_state.reminder_data['repeat_type']} reminder for {st.session_state.reminder_data['desc']} on {final_dt_obj.strftime('%A, %B %d at %I:%M %p')}.",
                        force_speak_st=True,
                    )
                )  # Confirmation message
        except Exception as e:
            print(f"Error setting reminder: {e}")
            responses.extend(
                generate_assistant_response_audio(
                    f"Sorry, there was an error setting the reminder: {e}",
                    force_speak_st=True,
                )
            )

        st.session_state.reminder_flow_step = None  # End flow
        st.session_state.reminder_data = {}  # Clear reminder data

    elif step == "remove_prompt":
        id_to_remove_str = action.strip().lower()
        removed_flag = False  # Reminder ID or description
        reminders = st.session_state.get(
            "reminders_for_removal", []
        )  # Get reminders for removal

        if (
            not id_to_remove_str or id_to_remove_str == "unintelligible"
        ):  # No selection made
            responses.extend(
                generate_assistant_response_audio(
                    "No selection made for removal. Cancelling.",
                    force_speak_st=True,
                )
            )
        elif id_to_remove_str.isdigit():  # If input is a digit (reminder ID)
            target_id = int(id_to_remove_str)
            if any(
                r[0] == target_id for r in reminders
            ):  # Check if ID exists
                if remove_reminder_db(target_id):  # Remove from database
                    responses.extend(
                        generate_assistant_response_audio(
                            f"Reminder number {target_id} removed.",
                            force_speak_st=True,
                        )
                    )
                    removed_flag = True
            else:
                responses.extend(
                    generate_assistant_response_audio(
                        f"No reminder found with number {target_id}.",
                        force_speak_st=True,
                    )
                )

        if not removed_flag and not id_to_remove_str.isdigit():  # If not removed by ID, try description
            matched_reminders = [
                r
                for r_id_match, desc_db, _, _ in reminders
                if id_to_remove_str in desc_db.lower()
                for r in [(r_id_match, desc_db)]
            ]  # Match by description
            if not matched_reminders:  # No match
                responses.extend(
                    generate_assistant_response_audio(
                        f"Could not find a reminder matching '{action}'. Try the number.",
                        force_speak_st=True,
                    )
                )
            elif len(matched_reminders) == 1:  # Single match
                r_id_match, desc_match = matched_reminders[0]
                st.session_state.reminder_flow_step = (
                    "remove_confirm"
                )  # Confirm removal
                st.session_state.reminder_to_confirm_remove = {
                    "id": r_id_match,
                    "desc": desc_match,
                }  # Store reminder for confirmation
                responses.extend(
                    generate_assistant_response_audio(
                        f"Found: Number {r_id_match}, '{desc_match}'. Remove it? (yes/no)",
                        force_speak_st=True,
                    )
                )
            else:  # Multiple matches
                responses.extend(
                    generate_assistant_response_audio(
                        "Multiple reminders match that description. Please specify by number:",
                        force_speak_st=True,
                    )
                )
                for r_id_multi, desc_multi in matched_reminders:  # List matches
                    responses.extend(
                        generate_assistant_response_audio(
                            f"Number {r_id_multi}: {desc_multi}",
                            force_speak_st=True,
                        )
                    )

        if removed_flag or (
            not matched_reminders and not id_to_remove_str.isdigit()
        ):  # If removed or no match
            st.session_state.reminder_flow_step = None  # End flow
            st.session_state.reminders_for_removal = None  # Clear stored reminders
            st.session_state.reminder_to_confirm_remove = (
                None
            )  # Clear reminder to confirm

    elif step == "remove_confirm":
        confirm_action = action.lower()
        r_data = st.session_state.get(
            "reminder_to_confirm_remove"
        )  # Get reminder data for confirmation
        if r_data and confirm_action in [
            "yes",
            "yeah",
            "ok",
            "yep",
        ]:  # If confirmation is yes
            if remove_reminder_db(r_data["id"]):  # Remove from database
                responses.extend(
                    generate_assistant_response_audio(
                        f"Reminder '{r_data['desc']}' removed.",
                        force_speak_st=True,
                    )
                )
            else:  # Should not happen if found earlier
                responses.extend(
                    generate_assistant_response_audio(
                        f"Error removing reminder '{r_data['desc']}'.",
                        force_speak_st=True,
                    )
                )
        elif r_data:  # Confirmation is no
            responses.extend(
                generate_assistant_response_audio(
                    f"Okay, reminder '{r_data['desc']}' was not removed.",
                    force_speak_st=True,
                )
            )
        else:  # Should not happen
            responses.extend(
                generate_assistant_response_audio(
                    "Error in removal confirmation. Please try again.",
                    force_speak_st=True,
                )
            )
        st.session_state.reminder_flow_step = None  # End flow
        st.session_state.reminders_for_removal = (
            None
        )  # Clear stored reminders
        st.session_state.reminder_to_confirm_remove = (
            None
        )  # Clear reminder to confirm

    if (
        not responses and st.session_state.reminder_flow_step is not None
    ):  # If a step was processed but no response generated yet
        pass  # Waiting for next user input for that step

    return responses


def get_weather_info(city_name):
    """Gets and returns weather information for a city."""
    responses = []
    st.session_state.last_assistant_question_context = (
        None
    )  # Clear previous context
    if (
        OPENWEATHERMAP_API_KEY == "YOUR_ACTUAL_OPENWEATHERMAP_API_KEY"
        or not OPENWEATHERMAP_API_KEY
        or len(OPENWEATHERMAP_API_KEY) < 30
    ):  # Check if API key is set
        msg = "Weather service is not configured correctly. Please add a valid OpenWeatherMap API key."
        print(msg)
        responses.extend(generate_assistant_response_audio(msg, force_speak_st=True))
        return responses

    params = {
        "q": city_name.strip(),
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric",
    }  # API request parameters
    try:
        response = requests.get(
            "http://api.openweathermap.org/data/2.5/weather?",
            params=params,
            timeout=10,
        )  # Make API request
        response.raise_for_status()
        weather_data = response.json()  # Raise HTTPError for bad responses
        if (
            str(weather_data.get("cod")) == "404"
        ):  # If city not found
            responses.extend(
                generate_assistant_response_audio(
                    f"Sorry, {weather_data.get('message', 'city not found')}.",
                    force_speak_st=True,
                )
            )
        elif (
            weather_data.get("main") and weather_data.get("weather")
        ):  # If data is valid
            main, weather = weather_data["main"], weather_data["weather"][0]
            wind_spd = (
                f"{weather_data.get('wind', {}).get('speed', 0) * 3.6:.1f} km/h"
            )
            report = f"In {weather_data.get('name', city_name.capitalize())}: Temperature is {main.get('temp')}° Celsius, feels like {main.get('feels_like')}° Celsius. The sky is {weather.get('description')}. Humidity is {main.get('humidity')} percent. Wind speed is {wind_spd}."  # Weather report
            responses.extend(
                generate_assistant_response_audio(report, force_speak_st=True)
            )
        else:
            responses.extend(
                generate_assistant_response_audio(
                    f"I received an unexpected weather response for {city_name}.",
                    force_speak_st=True,
                )
            )
    except requests.exceptions.HTTPError as http_err:
        print(f"Weather HTTP error: {http_err}")
        if http_err.response.status_code == 401:  # Invalid API key
            responses.extend(
                generate_assistant_response_audio(
                    "The weather API key is invalid or not authorized.",
                    force_speak_st=True,
                )
            )
        else:
            responses.extend(
                generate_assistant_response_audio(
                    f"There's a weather service error (code {http_err.response.status_code}).",
                    force_speak_st=True,
                )
            )
    except requests.exceptions.RequestException as e:
        print(f"Weather request error: {e}")
        responses.extend(
            generate_assistant_response_audio(
                "I can't connect to the weather service right now.",
                force_speak_st=True,
            )
        )
    except Exception as e:
        print(f"Weather processing error: {e}")
        responses.extend(
            generate_assistant_response_audio(
                "Sorry, an error occurred while fetching weather information.",
                force_speak_st=True,
            )
        )
    return responses


def search_youtube(query):
    """Searches YouTube for a video and returns the URL and title."""
    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "default_search": "ytsearch1:",
        "extract_flat": "discard_in_playlist",
        "forcejson": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # Use yt_dlp for YouTube searches
            result = ydl.extract_info(query, download=False)
            if result and "entries" in result and result["entries"]:
                return (
                    result["entries"][0].get("url"),
                    result["entries"][0].get("title", "Unknown Title"),
                )
            elif result and "webpage_url" in result:
                return (
                    result.get("webpage_url"),
                    result.get("title", "Unknown Title"),
                )
        print(f"No YouTube video for: {query}")
        return None, None
    except yt_dlp.utils.DownloadError as e:
        print(f"yt-dlp DL error for '{query}': {e}")
        return None, None
    except Exception as e:
        print(f"YouTube search error for '{query}': {e}")
        return None, None


def get_best_audio_url(video_url):
    """Gets the best audio URL from a YouTube video URL."""
    if not video_url:
        return None
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
        "quiet": True,
        "noplaylist": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # Use yt_dlp for audio extraction
            info = ydl.extract_info(video_url, download=False)
            if (
                "url" in info
                and info["url"].startswith("http")
                and not info.get("formats")
            ):
                return info["url"]
            if "formats" in info:
                audio_formats = [
                    f
                    for f in info["formats"]
                    if f.get("acodec") != "none"
                    and f.get("vcodec") == "none"
                    and f.get("url")
                ]
                if not audio_formats:
                    audio_formats = [
                        f for f in info["formats"] if f.get("acodec") != "none" and f.get("url")
                    ]
                if audio_formats:
                    for ext_pref in ["m4a", "webm", "ogg", "mp3"]:
                        for f_format in audio_formats:
                            if f_format.get("ext") == ext_pref:
                                return f_format["url"]
                    return audio_formats[0]["url"]
            return None
    except yt_dlp.utils.DownloadError as e_yt_dlp_dl_audio:
        print(
            f"yt-dlp DownloadError getting audio URL for '{video_url}': {e_yt_dlp_dl_audio}"
        )
        return None
    except Exception as e_audio_url:
        print(f"Error getting audio URL for '{video_url}': {e_audio_url}")
        return None


def play_song_st(song_query):
    """Plays a song from YouTube in Streamlit."""
    responses = []
    st.session_state.is_music_playing_st = (
        False  # Stop any previous music
    )
    st.session_state.current_music_url_st = (
        None  # Clear current music URL
    )
    st.session_state.current_music_title_st = (
        None  # Clear current music title
    )

    responses.extend(
        speak_random_st(
            [f"Looking for {song_query}...", f"Finding {song_query}..."],
            force_speak_st=True,
        )
    )  # Generate "looking for" message

    video_url, video_title = search_youtube(
        song_query
    )  # Search YouTube for the song
    if not video_url:
        responses.extend(
            generate_assistant_response_audio(
                f"Sorry, I couldn't find '{song_query}' on YouTube.",
                force_speak_st=True,
            )
        )  # Generate "not found" message
        return responses

    responses.extend(
        generate_assistant_response_audio(
            f"Found '{video_title}'. Getting the audio stream...", force_speak_st=True
        )
    )  # Generate "found" message

    playurl = get_best_audio_url(video_url)  # Get best audio URL
    if not playurl:
        responses.extend(
            generate_assistant_response_audio(
                f"I found '{video_title}', but I'm having trouble getting a playable audio stream for it.",
                force_speak_st=True,
            )
        )  # Generate "trouble getting audio" message
        return responses

    print(f"Streamlit: Attempting to play URL: {playurl}")
    st.session_state.current_music_url_st = playurl  # Set the music URL in session state
    st.session_state.current_music_title_st = (
        video_title
    )  # Set the music title in session state
    st.session_state.is_music_playing_st = (
        True  # Set the music playing flag in session state
    )
    responses.extend(
        generate_assistant_response_audio(
            f"Now playing: {video_title}.", force_speak_st=True
        )
    )  # Generate "now playing" message

    return responses


def stop_song_st():
    """Stops the currently playing song in Streamlit."""
    responses = []
    action_taken = False
    if st.session_state.get("is_music_playing_st", False):  # Check if music is playing
        action_taken = True

    st.session_state.is_music_playing_st = (
        False  # Set music playing flag to False
    )
    st.session_state.current_music_url_st = (
        None  # Clear current music URL
    )
    st.session_state.current_music_title_st = (
        None  # Clear current music title
    )

    response_text = (
        random.choice(["Music stopped.", "Playback halted."])
        if action_taken
        else random.choice(["No music playing.", "Nothing to stop."])
    )  # Response text
    responses.extend(
        generate_assistant_response_audio(response_text, force_speak_st=True)
    )  # Generate stop message
    return responses


def perform_web_search_and_summarize(query_text):
    """Performs a web search and summarizes the first result."""
    responses = []
    responses.extend(
        speak_random_st(
            [f"Looking up '{query_text}'...", f"Searching for '{query_text}'..."],
            force_speak_st=True,
        )
    )  # Generate "looking up" message
    search_results_list = []
    try:
        try:
            print(f"Attempting web search for: {query_text}")  # Log search attempt
            search_results_iter = search(
                query_text, num_results=1, lang="en", sleep_interval=1
            )  # Search the web
            search_results_list = list(search_results_iter)
            print(f"Search results found: {search_results_list}")  # Log results
        except Exception as e_search:
            print(f"Search Error: {e_search}")
            responses.extend(
                generate_assistant_response_audio(
                    f"I encountered a web search issue: {e_search}", force_speak_st=True
                )
            )
            return responses

        if not search_results_list:
            responses.extend(
                generate_assistant_response_audio(
                    f"I couldn't find any web results for '{query_text}'.",
                    force_speak_st=True,
                )
            )  # Generate "no results" message
            return responses

        first_url = next(
            (url for url in search_results_list if url.startswith("http")), None
        )  # Get the first URL
        if not first_url:
            responses.extend(
                generate_assistant_response_audio(
                    "Sorry, I couldn't find a valid web page to summarize.",
                    force_speak_st=True,
                )
            )  # Generate "no valid page" message
            return responses
        print(f"Summarizing: {first_url}")
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}  # Add User-Agent
            print(f"Attempting to get content from: {first_url}")  # Log content fetch
            try:
                response_req = requests.get(
                    first_url, headers=headers, timeout=15, verify=False, allow_redirects=True  # Disable SSL verification (temporary) and allow redirects
                )  # Get the page content
                response_req.raise_for_status()
                print(f"Content fetched successfully. Status code: {response_req.status_code}")  # Log status
            except requests.exceptions.RequestException as req_err:
                print(f"Request Error: {req_err}")
                responses.extend(generate_assistant_response_audio(f"I couldn't access the page. There was a request error: {req_err}", force_speak_st=True))
                return responses
            
            # Try different encodings
            try:
                print("Attempting to parse content with html.parser")
                soup = BeautifulSoup(response_req.content, "html.parser")
            except Exception as e_soup:
                print(f"Error parsing with html.parser: {e_soup}")
                try:
                    print("Attempting to parse content with utf-8 decoding")
                    soup = BeautifulSoup(response_req.content.decode('utf-8', 'ignore'), "html.parser")
                except Exception as e_utf8:
                    print(f"Error parsing with utf-8: {e_utf8}")
                    try:
                        print("Attempting to parse content with latin-1 decoding")
                        soup = BeautifulSoup(response_req.content.decode('latin-1', 'ignore'), "html.parser")
                    except Exception as e_latin1:
                        print(f"Error parsing with latin-1: {e_latin1}")
                        responses.extend(generate_assistant_response_audio("I couldn't parse the page content.", force_speak_st=True))
                        return responses
            
            for el_type in [
                "script",
                "style",
                "nav",
                "footer",
                "aside",
                "header",
                "form",
                "button",
                "img",
                "iframe",
                "link",
                "meta",
                "noscript",
                "a",
                "input",
                "select",
                "textarea",
                "figure",
                "figcaption",
                "svg",
                "path",
                "video",
                "audio",
                "canvas",
                "map",
                "area",
            ]:  # Remove irrelevant elements
                for el in soup.find_all(el_type):
                    el.decompose()
        
            paragraphs = [
                p.get_text(" ", strip=True)
                for p in soup.find_all("p")
                if p.get_text(strip=True) and len(p.get_text(strip=True).split()) > 10
            ]  # Extract paragraphs
            if paragraphs:
                full_text = " ".join(paragraphs)
                sentences = re.split(
                    r"(?<=[.!?])\s+(?=[A-Z" "'(])", full_text
                )  # Split into sentences
                summary_parts, wc, max_w = [], 0, 80
                for s_item in sentences:
                    if not s_item.strip():
                        continue
                    summary_parts.append(s_item.strip())
                    wc += len(s_item.split())
                    if wc > max_w:
                        break
                page_content_summary = (
                    " ".join(summary_parts)
                    if summary_parts
                    else "Could not extract meaningful summary from this page."
                )  # Create summary
            else:
                page_content_summary = "No suitable paragraphs found for summary on this page."
        except requests.exceptions.SSLError as ssl_err:  # Catch SSL-related errors
            print(f"SSL Error: {ssl_err}")
            page_content_summary = f"SSL Verification Failed: {ssl_err}"  # Customize message
        except requests.exceptions.RequestException as req_err:
            print(f"Request Error: {req_err}")
            page_content_summary = f"Request failed: {req_err}"
        except Exception as e:
            print(f"Summarization error for {first_url}: {e}")
            page_content_summary = f"I had an issue summarizing the page due to an error: {e}"
    except Exception as e:
        print(f"Outer Summarization error for {first_url}: {e}")
        page_content_summary = f"A major error occurred: {e}"

    final_summary_text = (
        f"Summary: {page_content_summary}"
        if page_content_summary
        and not page_content_summary.startswith("I had an issue") and not page_content_summary.startswith("SSL Verification Failed") and not page_content_summary.startswith("Request failed") and not page_content_summary.startswith("A major error occurred") and not page_content_summary.startswith("Could not extract meaningful summary from this page.") and not page_content_summary.startswith("No suitable paragraphs found for summary on this page.")
        else page_content_summary
    )  # Summary text
    responses.extend(
        generate_assistant_response_audio(
            final_summary_text, force_speak_st=True
        )
    )  # Generate summary message

    if final_summary_text and not final_summary_text.startswith("SSL Verification Failed") and not final_summary_text.startswith("Request failed") and not final_summary_text.startswith("A major error occurred") and not page_content_summary.startswith("Could not extract meaningful summary from this page.") and not page_content_summary.startswith("No suitable paragraphs found for summary on this page."):  # Skip question if summary failed
        responses.extend(
            generate_assistant_response_audio(
                "Would you like me to open this page for more details?",
                force_speak_st=True,
            )
        )  # Generate open page question
        st.session_state.last_assistant_question_context = {
            "type": "open_page_confirm",
            "url": first_url,
        }  # Set the "open page" context
    return responses

def web_search_detailed(query, num_results=1):
    """Performs a detailed web search and returns the results."""
    try:
        results = list(
            search(query, num_results=num_results, lang="en", sleep_interval=1)
        )  # Search the web
        results = [
            url for url in results if url.startswith("http")
        ]  # Filter for HTTP URLs
        return results
    except Exception as e:
        print(f"Detailed search error: {e}")
        return []


def get_page_content_detailed(url):
    """Gets detailed page content from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        try:
            response = requests.get(
                url, headers=headers, timeout=10, verify=False, allow_redirects=True
            )  # Get the page content
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error getting detailed content for {url}: {e}")
            return f"Could not fetch or process {url} due to a request error: {e}"

        try:
            try:
                soup = BeautifulSoup(response.text, "html.parser")
            except Exception as e_soup:
                try:
                    soup = BeautifulSoup(response.text.decode('utf-8', 'ignore'), "html.parser")
                except:
                    soup = BeautifulSoup(response.text.decode('latin-1', 'ignore'), "html.parser")
            for el in soup(
                [
                    "script",
                    "style",
                    "nav",
                    "footer",
                    "iframe",
                    "aside",
                    "header",
                    "form",
                    "button",
                    "img",
                    "a",
                ]
            ):
                el.decompose()  # Remove irrelevant elements
            texts = [
                el.get_text(" ", strip=True)
                for tag_name in ["article", "main", "section", "div", "p"]
                for el in soup.find_all(tag_name)
                if el.get_text(strip=True) and len(el.get_text(strip=True).split()) > 10
            ]  # Extract texts
            if texts:
                content = ". ".join(texts)
                sentences = [
                    s.strip() for s in content.split(".") if s.strip()
                ]  # Split into sentences
                keywords = [
                    "study",
                    "research",
                    "found",
                    "show",
                    "reveal",
                    "important",
                    "key",
                    "conclusion",
                ]  # Important keywords
                imp_sentences = sentences[:2]
                for s_item in sentences[2:8]:
                    if any(k in s_item.lower() for k in keywords) and len(
                        imp_sentences
                    ) < 5:
                        imp_sentences.append(s_item)
                return (
                    ". ".join(imp_sentences[:5]) + "."
                ).strip() if imp_sentences else "No specific important sentences found."  # Return important sentences
            return "No suitable content for detailed summary on this page."
        except Exception as e:
            print(f"Error getting detailed content for {url}: {e}")
            return f"An error occurred while fetching or processing content from this page: {url}. Error: {e}"
    except Exception as e:
        print(f"Outer Error getting detailed content for {url}: {e}")
        return f"A major error occurred while fetching or processing content from this page: {url}. Error: {e}"

def handle_detailed_web_search(query):
    """Handles detailed web searches and provides a summary."""
    responses = []
    responses.extend(
        generate_assistant_response_audio(
            f"Okay, performing a detailed search for {query}...",
            force_speak_st=True,
        )
    )  # Generate "searching" message
    results = web_search_detailed(query, num_results=1)  # Search the web
    if not results:
        responses.extend(
            generate_assistant_response_audio(
                "I didn't find any results for that detailed search.",
                force_speak_st=True,
            )
        )  # Generate "no results" message
        return responses

    top_url = results[0]
    print(f"\nDetailed search top URL: {top_url}")
    responses.extend(
        generate_assistant_response_audio(
            f"I found a page. Extracting detailed content now...", force_speak_st=True
        )
    )  # Generate "extracting" message
    content_summary = get_page_content_detailed(
        top_url
    )  # Get detailed summary

    if content_summary and not content_summary.startswith(
        "Could not"
    ) and not content_summary.startswith("An error") and not content_summary.startswith("No suitable content for detailed summary on this page."):
        print(f"\nDetailed Summary: {content_summary}\n")
        responses.extend(
            generate_assistant_response_audio(
                f"Here's the detailed information I found: {content_summary[:300]}...",
                force_speak_st=True,
            )
        )  # Generate summary message
        responses.extend(
            generate_assistant_response_audio(
                "Should I open the page for all the details?", force_speak_st=True
            )
        )  # Generate open page question
        st.session_state.last_assistant_question_context = {
            "type": "open_page_confirm",
            "url": top_url,
        }  # Set the "open page" context
    else:
        print(f"Detailed content extraction failed: {content_summary}")
        fail_message = f"Sorry, I {content_summary.lower() if content_summary else 'could not extract detailed content from this page.'}"
        responses.extend(
            generate_assistant_response_audio(fail_message, force_speak_st=True)
        )
    return responses


def predict_intent_from_text(text):
    """Predicts the intent from a given text using the trained model."""
    try:
        if not text or len(text.strip()) == 0:
            print("Empty text input for intent prediction.")
            return None, 0.0

        seq = tokenizer.texts_to_sequences([text])
        maxlen_model = 20

        # استخلاص الطول من نموذج Keras إن أمكن
        if hasattr(model, "input_shape") and isinstance(model.input_shape, (tuple, list)):
            if len(model.input_shape) > 1 and model.input_shape[1] is not None:
                maxlen_model = model.input_shape[1]

        padded = pad_sequences(seq, maxlen=maxlen_model, padding="post", truncating="post")
        prediction = model.predict(padded, verbose=0)

        tag_index = np.argmax(prediction[0])
        confidence = float(prediction[0][tag_index])

        tag = label_encoder.inverse_transform([tag_index])[0]
        return tag, confidence

    except Exception as e:
        print(f"Error in predict_intent_from_text: {e}")
        return None, 0.0

def handle_general_conversation_query(query_text):
    """Handles general conversation queries using intent recognition."""
    responses = []
    is_handled = False

    if not query_text or query_text.strip() == "":
        print("Empty query provided to handle_general_conversation_query.")
        return is_handled, responses

    tag, confidence = predict_intent_from_text(query_text) 
    
    if tag:
        print(f"Intent detected: {tag}, Confidence: {confidence:.3f}")

        # تعديل: التعامل مع النوايا ذات الثقة الأقل
        if confidence < 0.6:  # يمكنك تجربة قيم مختلفة هنا
            print("Confidence too low for intent handling. Attempting a general response.")
            # يمكنك هنا إضافة كود لمحاولة التعامل مع الأمر بشكل عام،
            # مثل البحث عن الكلمات المفتاحية في النص أو استخدام نموذج لغوي عام للإجابة.
            # في المثال الحالي، سنكتفي بإرجاع أن الأمر لم يتم فهمه.
            return is_handled, responses  # العودة بدون معالجة النية المحددة

        # التأكد من أن الرد لم يُرسل سابقًا
        last_response = ""
        if st.session_state.messages:
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant":
                    last_response = msg.get("content", "")
                    break

        # البحث عن النية المطابقة في intents.json
        for intent_data in data.get("intents", []):
            if intent_data["tag"].lower() == tag.lower():
                response_text = random.choice(intent_data["responses"])

                # فقط أضف الرد إن كان مختلفًا
                if response_text != last_response:
                    audio_responses = generate_assistant_response_audio(
                        response_text,
                        force_speak_st=tag in ["greetings", "thanks", "goodbye", "help"]
                    )
                    responses.extend(audio_responses)
                    is_handled = True
                else:
                    print(f"Duplicate response avoided: '{response_text}'")
                break
    else:
        print("No intent detected.")

    return is_handled, responses
# --- System Interaction Functions (SERVER-SIDE for Streamlit) ---
def wishMe_st():
    """Greets the user with a personalized message based on the time of day."""
    return generate_assistant_response_audio(
        f"Good {('Morning' if 0 <= datetime.datetime.now().hour < 12 else 'Afternoon' if 12 <= datetime.datetime.now().hour < 18 else 'Evening')}! I am your web assistant. How can I help?",
        force_speak_st=True,
    )


def send_email_st(recipient_email, email_content, user_email, user_password):
    """Sends an email using the provided credentials."""
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(user_email, user_password)

            message = MIMEMultipart()
            message['From'] = user_email
            message['To'] = recipient_email
            message['Subject'] = "Email from your personal assistant"
            message.attach(MIMEText(email_content, 'plain'))

            server.sendmail(user_email, recipient_email, message.as_string())
        return generate_assistant_response_audio("Email sent!", force_speak_st=True)
    except Exception as e:
        print(f"Email sending error: {e}")
        return generate_assistant_response_audio(f"Sorry, there was an error sending the email: {e}", force_speak_st=True)

def handle_email_sending_st(query, user_email, user_password):
    """Handles email sending in Streamlit after login."""
    responses = []
    st.session_state.last_assistant_question_context = None
    match = re.search(r"send an email to\s+(.+)", query, re.IGNORECASE)
    if not match:
        responses.extend(
            generate_assistant_response_audio(
                "To whom do you want to send the email?",
                force_speak_st=True,
            )
        )
        st.session_state.email_flow_step = "get_recipient"  # New state

    else:
        recipient_email = match.group(1).strip()
        responses.extend(
            generate_assistant_response_audio(
                f"Okay, what should the email say to {recipient_email}?",
                force_speak_st=True,
            )
        )
        st.session_state.email_recipient = recipient_email  # Store recipient
        st.session_state.email_flow_step = "get_content"  # New state
    return responses # Return here to allow new state to take effect

import requests
import logging
import os
import time  # Import time for sleep


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LOCATION_SERVICE_URL = os.environ.get("LOCATION_SERVICE_URL", "https://ipinfo.io/json")
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def get_location_st():
    """Gets the server's location, with retry mechanism."""
    responses = []
    for attempt in range(MAX_RETRIES):
        try:
            try:
                logging.info(f"Attempting to get location (attempt {attempt + 1})...")
                response = requests.get(LOCATION_SERVICE_URL, timeout=7)
                response.raise_for_status()
                data_loc = response.json()
                responses.extend(
                    generate_assistant_response_audio(
                        f"Based on the server's IP, it looks like it's near {data_loc.get('city', 'an unknown city')}, {data_loc.get('region', 'an unknown region')}.",
                        force_speak_st=True,
                    )
                )
                return responses  # Success! Exit loop
            except requests.exceptions.Timeout:
                logging.warning("Location service request timed out.")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)  # Wait before retrying
            except requests.exceptions.ConnectionError as e:
                logging.error(f"Location service connection error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
            except requests.exceptions.RequestException as e:
                logging.error(f"Location service request error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
            except (ValueError, KeyError) as e:
                logging.error(f"Location service data parsing error: {e}")
                responses.extend(
                    generate_assistant_response_audio(
                        "I received an invalid response from the location service.",
                        force_speak_st=True,
                    )
                )
                return responses  # No point retrying data parsing errors
            except Exception as e:
                logging.error(f"Unexpected location error: {e}", exc_info=True)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY) # Retry unless max attempts reached
            
            
        except Exception as outer_e:  # Outer Catch-All for any errors NOT request specific
            logging.error(f"Outer General Error occurred: {outer_e}", exc_info=True)
            if attempt < MAX_RETRIES - 1: # Retry unless max attempts reached
                time.sleep(RETRY_DELAY)
        
        
    # If we reach here, all retries failed
    responses.extend(
            generate_assistant_response_audio(
                "I couldn't determine the server's location after multiple attempts. There may be a persistent network issue.",
                force_speak_st=True,
            )
        )

    return responses

def translate_text_flow(text_to_translate, target_language_name):
    """Translates text to the specified language."""
    responses = []
    if not text_to_translate:  # If no text is provided
        responses.extend(
            generate_assistant_response_audio(
                "What text should I translate?", force_speak_st=True
            )
        )
        st.session_state.last_assistant_question_context = {
            "type": "get_text_for_translation",
            "target_lang_name": target_language_name,
        }  # Set context
        return responses

    target_lang_lower = target_language_name.lower().strip()  # Normalize target language
    if target_lang_lower not in language_map:  # If language is not supported
        supported_langs = ", ".join(list(language_map.keys())[:5])
        responses.extend(
            generate_assistant_response_audio(
                f"I can't translate to {target_language_name}. I support languages like: {supported_langs}...",
                force_speak_st=True,
            )
        )
        return responses

    target_lang_code = language_map[target_lang_lower]  # Get the language code
    translated_text, used_translator = None, "N/A"
    try:
        translated_text = GoogleTranslator(
            source="auto", target=target_lang_code
        ).translate(
            text_to_translate
        )  # Translate using DeepTranslator
        used_translator = "DeepTranslator"
    except Exception as e_deep:
        print(f"DeepTranslator error: {e_deep}. Trying fallback.")
        if google_translate_service_instance:  # If googletrans is available
            try:
                trans_obj = google_translate_service_instance.translate(
                    text_to_translate, dest=target_lang_code
                )  # Translate using googletrans
                translated_text = trans_obj.text
                used_translator = "googletrans"
            except Exception as e_trans:
                print(f"googletrans error: {e_trans}")

    if not translated_text:  # If translation failed
        responses.extend(
            generate_assistant_response_audio(
                f"I had some trouble translating to {target_language_name}.",
                force_speak_st=True,
            )
        )
        return responses

    responses.extend(
        generate_assistant_response_audio(
            f"In {target_language_name.capitalize()}, '{text_to_translate}' is: '{translated_text}'. (Translated using {used_translator}).",
            force_speak_st=True,
        )
    )  # Generate translation message
    if target_lang_code != "en":  # If not translating to English
        responses.extend(
            generate_assistant_response_audio(
                f"Would you like me to say that in {target_language_name.capitalize()}?",
                force_speak_st=True,
            )
        )  # Generate "say it" question
        st.session_state.last_assistant_question_context = {
            "type": "say_translation_confirm",
            "text": translated_text,
            "lang_code": target_lang_code,
            "lang_name": target_language_name.capitalize(),
        }  # Set context
    else:
        st.session_state.last_assistant_question_context = (
            None  # Clear context if translated to English
        )
    return responses


def open_social_media(platform_name):
    """Opens a social media platform in the server's browser."""
    responses = []
    urls = {
        "facebook": "https://facebook.com",
        "twitter": "https://twitter.com",
        "youtube": "https://youtube.com",
        "instagram": "https://instagram.com",
        "linkedin": "https://linkedin.com",
    }  # URLs for social media platforms
    url = urls.get(platform_name.lower().strip())  # Get URL
    if url:
        responses.extend(
            generate_assistant_response_audio(
                f"Opening {platform_name.capitalize()} on the server.",
                force_speak_st=True,
            )
        )  # Generate "opening" message
        webbrowser.open(url)  # Opens on server
    else:
        responses.extend(
            generate_assistant_response_audio(
                f"I don't have a link for {platform_name}.", force_speak_st=True
            )
        )  # Generate "no link" message
    return responses


def open_application_server(app_name_query):
    """Opens an application on the server."""
    responses = []
    global opened_processes  # This global is for server-side processes
    app_map = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "chrome": "chrome.exe",
    }  # Example for Windows
    cmd, name_app, exe_app = None, None, None
    query_lower = app_name_query.lower().strip()
    for n_app, c_app in app_map.items():  # Iterate over applications
        if n_app in query_lower:
            cmd, name_app, exe_app = c_app, n_app.capitalize(), c_app
            break
    if cmd:
        responses.extend(
            generate_assistant_response_audio(
                f"Attempting to open {name_app} on the server.", force_speak_st=True
            )
        )  # Generate "attempting" message
        try:
            proc = subprocess.Popen(
                cmd
            )  # Open application using subprocess
            opened_processes[exe_app.lower()] = proc  # Tracks server-side processes
            responses.extend(
                generate_assistant_response_audio(
                    f"{name_app} should be opening on the server.",
                    force_speak_st=True,
                )
            )  # Generate "should be opening" message
        except Exception as e:
            responses.extend(
                generate_assistant_response_audio(
                    f"Error opening {name_app} on the server: {e}",
                    force_speak_st=True,
                )
            )  # Generate "error" message
    else:
        responses.extend(
            generate_assistant_response_audio(
                f"I don't know how to open '{app_name_query}' on the server.",
                force_speak_st=True,
            )
        )  # Generate "don't know how" message
    return responses


def close_application_server(app_name_query):
    """Closes an application on the server."""
    responses = []
    global opened_processes
    query_lower = app_name_query.lower().strip()
    app_exe_map = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "chrome": "chrome.exe",
    }
    exe_to_close = next(
        (
            exe_map
            for name_map, exe_map in app_exe_map.items()
            if name_map in query_lower
        ),
        query_lower
        if query_lower.endswith(".exe")
        else query_lower + ".exe"
        if sys.platform == "win32"
        else query_lower,
    )

    closed_msg = ""
    if sys.platform == "win32" and exe_to_close:  # If Windows
        try:
            res = subprocess.run(
                ["taskkill", "/F", "/IM", exe_to_close],
                check=True,
                capture_output=True,
                text=True,
            )  # Use taskkill command
            if "SUCCESS" in res.stdout:
                closed_msg = f"Closed {app_name_query} on the server."
                if exe_to_close.lower() in opened_processes:
                    opened_processes.pop(
                        exe_to_close.lower()
                    )  # Remove from opened processes
            else:
                closed_msg = f"Could not find or close {app_name_query} on the server. Taskkill output: {res.stderr or res.stdout}"
        except Exception as e:
            closed_msg = f"Error closing {app_name_query} on the server: {e}"
    elif exe_to_close.lower() in opened_processes:  # If not Windows
        proc = opened_processes.pop(exe_to_close.lower())
        proc.terminate()  # Terminate the process
        try:
            proc.wait(timeout=2)
            closed_msg = f"Closed {app_name_query} on the server."
        except subprocess.TimeoutExpired:  # If termination timed out
            proc.kill()  # Kill the process
            closed_msg = f"Force closed {app_name_query} on the server."
        except Exception as e:
            closed_msg = f"Error terminating {app_name_query} on server: {e}"
    else:
        closed_msg = f"I don't know how to close '{app_name_query}' on the server, or it wasn't opened by me."
    responses.extend(
        generate_assistant_response_audio(closed_msg, force_speak_st=True)
    )  # Generate close message
    return responses


def perform_browsing_server(search_term):
    """Performs a web search on the server's browser."""
    responses = []
    try:
        webbrowser.open(
            f"https://www.google.com/search?q={requests.utils.quote(search_term.strip())}"
        )  # Open the search URL in the server's default browser
        responses.extend(
            generate_assistant_response_audio(
                f"Searching Google for '{search_term}' on the server's browser.",
                force_speak_st=True,
            )
        )  # Generate "searching" message
    except Exception as e:
        responses.extend(
            generate_assistant_response_audio(
                f"Error opening search on the server: {e}", force_speak_st=True
            )
        )  # Generate "error" message
    return responses


def check_system_condition_server():
    """Checks the server's system condition (CPU, memory, disk usage)."""
    responses = []
    try:
        cpu = psutil.cpu_percent(interval=0.5)  # CPU usage
        mem = psutil.virtual_memory()  # Memory info
        disk_path = "/"
        if sys.platform == "win32":  # Windows specific disk path
            system_root = os.environ.get("SystemRoot", "C:\\")
            disk_path = os.path.splitdrive(system_root)[0] + os.sep

        disk_report_str = "Could not retrieve disk usage information for the server."
        try:
            disk = psutil.disk_usage(disk_path)  # Disk usage
            disk_report_str = (
                f"Server disk space ({disk_path}) is {disk.percent}% used "
                f"({disk.free / 1024 ** 3:.1f}GB free)."
            )
        except Exception as e_disk:
            print(f"Error getting server disk usage for '{disk_path}': {e_disk}")

        report_parts = [
            f"Server CPU is at {cpu}%",
            f"Server memory is at {mem.percent}% used ({mem.available / 1024 ** 3:.1f}GB free)",
            disk_report_str,
        ]

        if hasattr(psutil, "sensors_battery"):  # Battery info (if available)
            battery = psutil.sensors_battery()
            if battery:
                plugged = (
                    "charging"
                    if battery.power_plugged and battery.percent < 100
                    else (
                        "full"
                        if battery.power_plugged and battery.percent >= 99.5
                        else "on battery"
                    )
                )
                report_parts.append(f"Server battery is at {battery.percent}% ({plugged})")
                if (
                    not battery.power_plugged
                    and battery.secsleft
                    not in (
                        psutil.POWER_TIME_UNLIMITED,
                        psutil.POWER_TIME_UNKNOWN,
                        None,
                    )
                    and battery.secsleft > 0
                ):
                    hours, remainder = divmod(battery.secsleft, 3600)
                    minutes, _ = divmod(remainder, 60)
                    report_parts.append(
                        f"About {int(hours)} hours and {int(minutes)} minutes remaining on server battery"
                    )

        responses.extend(
            generate_assistant_response_audio(
                "Server System Status: " + ". ".join(report_parts) + ".",
                force_speak_st=True,
            )
        )  # Generate system status message

    except Exception as e_psutil:
        print(f"Error getting server system condition with psutil: {e_psutil}")
        responses.extend(
            generate_assistant_response_audio(
                "I'm having trouble getting the full server system status at the moment.",
                force_speak_st=True,
            )
        )
    return responses


def change_system_volume_server(direction, amount=10):
    """Changes the server's system volume."""
    responses = []
    msg = ""
    try:
        if sys.platform == "win32":  # Windows volume control
            if direction == "up":
                pyautogui.press(
                    "volumeup", presses=amount // 5 if amount > 5 else 1
                )
            elif direction == "down":
                pyautogui.press(
                    "volumedown", presses=amount // 5 if amount > 5 else 1
                )
            elif direction in ["mute", "unmute"]:
                pyautogui.press("volumemute")  # Toggles
            msg = f"Server volume {direction}."
        elif sys.platform == "darwin":  # macOS volume control
            script = ""
            if direction == "up":
                script = (
                    "set volume output volume (output volume of (get volume settings) +"
                    f" {amount})"
                )
            elif direction == "down":
                script = (
                    "set volume output volume (output volume of (get volume settings) -"
                    f" {amount})"
                )
            elif direction == "mute":
                script = "set volume output muted true"
            elif direction == "unmute":
                script = "set volume output muted false"
            if script:
                subprocess.run(["osascript", "-e", script], check=True)
            msg = f"Server volume {direction}."
        elif sys.platform.startswith("linux"):  # Linux volume control
            current_vol_cmd = ["amixer", "sget", "Master"]
            process = subprocess.run(
                current_vol_cmd, capture_output=True, text=True, check=True
            )
            match = re.search(r"\[(\d+)%\]", process.stdout)
            current_percentage = int(match.group(1)) if match else 50

            new_vol = current_percentage
            if direction == "up":
                new_vol = min(100, current_percentage + amount)
            elif direction == "down":
                new_vol = max(0, current_percentage - amount)
            elif direction == "mute":
                subprocess.run(["amixer", "-q", "sset", "Master", "mute"], check=True)
                msg = "Server volume muted."
            elif direction == "unmute":
                subprocess.run(["amixer", "-q", "sset", "Master", "unmute"], check=True)
                msg = "Server volume unmuted."

            if direction in ["up", "down"]:
                subprocess.run(
                    ["amixer", "-q", "sset", "Master", f"{new_vol}%"],
                    check=True,
                )
                msg = f"Server volume set to {new_vol} percent."
        else:
            msg = "Server volume control not supported on this OS yet."
    except Exception as e:
        print(f"Error changing server volume: {e}")
        msg = f"Could not change server system volume: {e}"

    if msg:
        responses.extend(
            generate_assistant_response_audio(msg, force_speak_st=True)
        )  # Generate volume message
    return responses


# --- Main Processing Function for Streamlit ---
def process_command_logic_st(query):
    responses = []
    processed_query = preprocess_input(query)
    print(f"Original Query: '{query}'")
    print(f"Processed Query: '{processed_query}'")

    is_convo_handled, convo_responses = handle_general_conversation_query(processed_query)
    if is_convo_handled:
        responses.extend(convo_responses)
        return responses
    
    tag, confidence = predict_intent_from_text(processed_query)
    print(f"Intent Prediction - Tag: '{tag}', Confidence: {confidence:.3f}'")
    if tag:
        if tag == "email_send":
            # استخراج البريد الإلكتروني من الأمر
            match = re.search(r"send an email to\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", query, re.IGNORECASE)
            if match:
                recipient_email = match.group(1)
                # التعامل مع إرسال البريد الإلكتروني هنا
                responses.extend(handle_email_sending_st(query, st.session_state.user_email, st.session_state.user_password))
                return responses
            else:
                responses.extend(generate_assistant_response_audio("Sorry, I couldn't find the email address.", force_speak_st=True))
                return responses
        
        if st.session_state.get("email_flow_step"): # Email
            if st.session_state.email_flow_step == "get_recipient":
                st.session_state.email_recipient = query.strip()
                responses.extend(
                    generate_assistant_response_audio(
                        f"Okay, what should the email say to {st.session_state.email_recipient}?",
                        force_speak_st=True,
                    )
                )
                st.session_state.email_flow_step = "get_content"
                return responses
            elif st.session_state.email_flow_step == "get_content":
                responses.extend(send_email_st(st.session_state.email_recipient, query.strip(), st.session_state.user_email, st.session_state.user_password))
                st.session_state.email_flow_step = None  # Email flow over
                return responses

        
    

        context_handled_this_turn = False  # Flag for context handling
        lc_query = query.lower().strip()  # Normalize query

        if st.session_state.get(
            "reminder_flow_step"
        ):  # Reminder flow takes priority
            responses.extend(
                manage_reminders_flow_st(query)
            )  # query is the detail user provided
            return responses  # End processing here, reminder flow handles it

        if st.session_state.get(
            "last_assistant_question_context"
        ):  # Handle context-based follow-up questions
            context = st.session_state.last_assistant_question_context  # Get context
            context_type = context.get("type")  # Context type
            positive_responses_list = [
                "yes",
                "sure",
                "okay",
                "do it",
                "yeah",
                "yep",
                "please",
                "confirm",
                "affirmative",
                "go ahead", 
                "open",
            ]  # Positive responses
            negative_responses_list = [
                "no",
                "nope",
                "don't",
                "cancel",
                "negative",
                "stop",
                "don't do it",
            ]  # Negative responses

            if context_type == "web_search_confirm":  # Web search confirmation context
                if any(
                    word in lc_query for word in positive_responses_list
                ):
                    responses.extend(
                        generate_assistant_response_audio(
                            f"Okay, searching the web for '{context['query']}'.",
                            force_speak_st=True,
                        )
                    )  # Generate "searching" message
                    responses.extend(
                        perform_web_search_and_summarize(context["query"])
                    )  # Perform search and summarize
                elif any(word in lc_query for word in negative_responses_list):
                    responses.extend(
                        generate_assistant_response_audio(
                            "Alright, I won't search.", force_speak_st=True
                        )
                    )  # Generate "won't search" message
                else:  # Unclear response, assume no and clear context
                    responses.extend(
                        generate_assistant_response_audio(
                            f"I'm not sure what to do about searching '{context['query']}'. I'll skip it for now.",
                            force_speak_st=True,
                        )
                    )
                st.session_state.last_assistant_question_context = None
                context_handled_this_turn = True  # Clear context

            elif context_type == "open_page_confirm":  # Open page confirmation context
                if any(
                    word in lc_query for word in positive_responses_list
                ):
                    responses.extend(
                        speak_random_st(
                            ["Opening the page on the server.", "Alright, opening it on the server."],
                            force_speak_st=True,
                        )
                    )  # Generate "opening" message
                    try:
                        webbrowser.open(context["url"])  # Opens on server
                    except Exception as e:
                        print(f"Error opening browser on server: {e}")
                elif any(word in lc_query for word in negative_responses_list):
                    responses.extend(
                        generate_assistant_response_audio(
                            "Okay, I won't open it.", force_speak_st=True
                        )
                    )  # Generate "won't open" message
                else:
                    responses.extend(
                        generate_assistant_response_audio(
                            "I'll assume you don't want to open the page for now.",
                            force_speak_st=True,
                        )
                    )
                st.session_state.last_assistant_question_context = None
                context_handled_this_turn = True  # Clear context

            elif context_type == "say_translation_confirm":  # Say translation confirmation context
                if any(
                    word in lc_query for word in positive_responses_list + ["say it"]
                ):
                    responses.extend(
                        generate_assistant_response_audio(
                            context["text"],
                            lang_code=context["lang_code"],
                            force_speak_st=True,
                        )
                    )  # Generate audio in translated language
                st.session_state.last_assistant_question_context = None
                context_handled_this_turn = True  # Clear context

            elif context_type == "get_city_for_weather":  # Get city for weather context
                if query and query != "unintelligible":
                    responses.extend(
                        get_weather_info(query.strip())
                    )  # Get weather info
                else:
                    responses.extend(
                        generate_assistant_response_audio(
                            "I didn't get the city for the weather. Please try again.",
                            force_speak_st=True,
                        )
                    )
                st.session_state.last_assistant_question_context = None
                context_handled_this_turn = True  # Clear context

            elif context_type == "get_song_for_play":  # Get song for play context
                if query and query != "unintelligible":
                    responses.extend(
                        play_song_st(query.strip())
                    )  # Play the song
                else:
                    responses.extend(
                        generate_assistant_response_audio(
                            "I didn't get the song name. Please try again.",
                            force_speak_st=True,
                        )
                    )
                st.session_state.last_assistant_question_context = None
                context_handled_this_turn = True  # Clear context

            elif context_type == "get_text_for_translation":  # Get text for translation context
                if query and query != "unintelligible":
                    responses.extend(
                        translate_text_flow(
                            query.strip(), context["target_lang_name"]
                        )
                    )  # Translate the text
                else:
                    responses.extend(
                        generate_assistant_response_audio(
                            f"I didn't get what you want to translate to {context['target_lang_name']}. Please try again.",
                            force_speak_st=True,
                        )
                    )
                st.session_state.last_assistant_question_context = None
                context_handled_this_turn = True  # Clear context

            if context_handled_this_turn:
                return responses  # If context is handled, return

        play_match = re.search(
            r"(?:play|listen to|put on|stream)\s+(?:the\s+)?(?:song\s+|music\s+|track\s+)?(.+)",
            query,
            re.IGNORECASE,
        )
        if play_match:
            st.session_state.last_assistant_question_context = None
            song_name = play_match.group(1).strip()
            if song_name:
                responses.extend(play_song_st(song_name))
            else:
                responses.extend(
                    generate_assistant_response_audio(
                        "Sure, what song or artist would you like to hear?",
                        force_speak_st=True,
                    )
                )
                st.session_state.last_assistant_question_context = {
                    "type": "get_song_for_play"
                }
            return responses
        elif re.search(
            r"stop\s*(?:the\s*)?(song|music|playback|playing)|stop current track",
            query,
            re.IGNORECASE,
        ):
            st.session_state.last_assistant_question_context = None
            responses.extend(stop_song_st())
            return responses
        # Pause, Next, Resume for st.audio are client-side controls, not easily Python-driven.
        # We can only stop/start a new stream.

        elif (
            m := re.search(
                r"(?:weather|forecast)(?:\s+(?:in|for|at|like in)\s+([a-zA-Z\s\-,'.]+))?",
                query,
                re.IGNORECASE,
            )
        ):
            st.session_state.last_assistant_question_context = None
            city = m.group(1).strip() if m.group(1) else None
            if not city:
                responses.extend(
                    generate_assistant_response_audio(
                        "For which city would you like the weather forecast?",
                        force_speak_st=True,
                    )
                )
                st.session_state.last_assistant_question_context = {
                    "type": "get_city_for_weather"
                }
            else:
                responses.extend(get_weather_info(city))
            return responses

        elif "reminder" in query:
            st.session_state.last_assistant_question_context = None
            st.session_state.reminder_flow_step = "action_prompted"
            st.session_state.reminder_data = {}
            responses.extend(manage_reminders_flow_st(None))
            return responses
        
        elif (
                m := re.search(
                    r"(?:send|email|mail)\s*(?:an)?\s*(?:email|mail)?\s*(?:to)?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})?",
                    query,
                    re.IGNORECASE,
                )
            ):
            if st.session_state.get("user_email") and st.session_state.get("user_password"):
                responses.extend(handle_email_sending_st(query, st.session_state.user_email, st.session_state.user_password))
                return responses
            else:
                st.session_state.email_login_needed = True
                responses.append(
                    {
                        "role": "assistant",
                        "text_content": "Please log in to send emails.",
                        "audio_path": None
                    })
                return responses


        elif re.search(r"where\s+am\s+i|my\s+location", query, re.IGNORECASE):
            st.session_state.last_assistant_question_context = None
            responses.extend(get_location_st())
            return responses
        elif re.search(
            r"system\s+(status|condition|info)|computer health", query, re.IGNORECASE
        ):
            st.session_state.last_assistant_question_context = None
            responses.extend(check_system_condition_server())
            return responses

        elif (
            m := re.search(
                r"translate\s*(?:this|the phrase)?\s*['\"]?(.*?)['\"]?\s+to\s+([a-zA-Z\s\-]+)",
                query,
                re.IGNORECASE,
            )
        ):
            st.session_state.last_assistant_question_context = None
            responses.extend(translate_text_flow(m.group(1).strip(), m.group(2).strip()))
            return responses

        elif (m := re.search(r"open\s+(.+)", query, re.IGNORECASE)):
            st.session_state.last_assistant_question_context = None
            target = m.group(1).strip().lower()
            social_platforms = {
                "facebook": 1,
                "twitter": 1,
                "youtube": 1,
                "instagram": 1,
                "linkedin": 1,
            }
            opened_flag = False
            for s_key in social_platforms:
                if s_key in target:
                    responses.extend(open_social_media(s_key))
                    opened_flag = True
                    break
            if not opened_flag:
                app_to_open = re.sub(
                    r"^(app|program|application)\s+", "", target, flags=re.I
                ).strip()
                responses.extend(open_application_server(app_to_open))
            return responses

        elif (m := re.search(r"close\s+(.+)", query, re.IGNORECASE)):
            st.session_state.last_assistant_question_context = None
            responses.extend(close_application_server(m.group(1).strip()))
            return responses

        elif "increase volume" in query or "volume up" in query or "Volume up" in query or "louder" in query or "make it louder" in query or "turn up the volume" in query or "Increase volume" in query:
            st.session_state.last_assistant_question_context = None
            responses.extend(change_system_volume_server("up"))
            return responses
        elif "decrease volume" in query or "volume down" in query or "quieter" in query or "make it quieter" in query or "turn down the volume" in query or "Lower volume" in query or "reduce volume" in query:
            st.session_state.last_assistant_question_context = None
            responses.extend(change_system_volume_server("down"))
            return responses
        elif "mute" in query and "volume" in query or "Mute" in query and "volume" in query :  # Ensure "volume" to avoid muting mic if assistant is named "Mute"
            st.session_state.last_assistant_question_context = None
            responses.extend(change_system_volume_server("mute"))
            return responses
        elif "unmute" in query and "volume" in query or "Unmute" in query and "volume" in query:  # Ensure "volume" to avoid unmuting mic if assistant is named "Mute"
            st.session_state.last_assistant_question_context = None
            responses.extend(change_system_volume_server("unmute"))
            return responses

        elif (
            m := re.search(r"(?:google|search google for)\s+(.+)", query, re.IGNORECASE
            )
        ):
            st.session_state.last_assistant_question_context = None
            responses.extend(perform_browsing_server(m.group(1).strip()))
            return responses
        elif (
            m := re.search(
                r"(?:detailed search for|scrape|extract about |what is|who is)\s+(.+)",
                query,
                re.IGNORECASE,
            )
        ):
            st.session_state.last_assistant_question_context = None
            responses.extend(handle_detailed_web_search(m.group(1).strip()))
            return responses
        elif (
            m := re.search(
                r"(?:summarize|tell me about  )\s+(.+?)(?:\s+on web)?$",
                query,
                re.IGNORECASE,
            )
        ):
            st.session_state.last_assistant_question_context = None
            responses.extend(perform_web_search_and_summarize(m.group(1).strip()))
            return responses
        
        elif "open page" in query or "open website" in query:
            st.session_state.last_assistant_question_context = None
            m = re.search(r"open\s+page\s+(.+)", query, re.IGNORECASE)
            if m:
                url = m.group(1).strip()
                if not url.startswith(("http://", "https://")):
                    url = "http://" + url
                responses.extend(
                    generate_assistant_response_audio(
                        f"Do you want me to open the page '{url}' on the server?",
                        force_speak_st=True,
                    )
                )
        
        elif "generate image" in query or "create image" in query or "make image" in query or "draw image" in query or "generate picture" in query or "create picture" in query or "make picture" in query or "draw picture" in query or "generate art" in query or "create art" in query or "make art" in query or "draw art" in query:
            st.session_state.last_assistant_question_context = None
            m = re.search(r"generate\s+image\s+(.+)", query, re.IGNORECASE)
            if m:
                prompt = m.group(1).strip()
                if prompt:
                    st.session_state.messages.append({"role": "user", "content": f"Generated Image prompt: {prompt}", "source": "text"})
                    with st.spinner(f"Generating image for: {prompt}"):
                        images = generate_image(prompt, service="stability")  # Online service
                        if images:
                            for img in images:
                                st.session_state.generated_images.append({"image": img, "caption": prompt})
                        else:
                            st.error("Image generation failed.")
                
                    st.session_state.last_assistant_question_context = {
                        "type": "image_generation",
                        "prompt": prompt,
                    }
                else:
                    responses.extend(
                        generate_assistant_response_audio(
                            "What should the image be about?",
                            force_speak_st=True,
                        )
                    )
                    st.session_state.last_assistant_question_context = {
                        "type": "get_image_prompt"
                    }
                return responses

        elif "screenshot" in query or "capture screen" in query or "take screenshot" in query or "capture screenshot" in query :
            st.session_state.last_assistant_question_context = None
            try:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                ss_dir = os.path.join(script_dir, "screenshots_st")
                os.makedirs(ss_dir, exist_ok=True)
                fn = os.path.join(ss_dir, f"ss_{ts}.png")
                pyautogui.screenshot(fn)
                responses.extend(
                    generate_assistant_response_audio(
                        f"Screenshot saved on the server in '{ss_dir}'.",
                        force_speak_st=True,
                    )
                )
            except Exception as e:
                print(f"Screenshot error on server: {e}")
                responses.extend(
                    generate_assistant_response_audio(
                        "Error taking screenshot on the server.", force_speak_st=True
                    )
                )
            return responses

        elif re.search(r"\b(exit|quit|goodbye|bye|terminate)\b", query, re.IGNORECASE):
            responses.extend(
                speak_random_st(["Goodbye!", "See you later!", "Shutting down this session."])
            )
            st.session_state.clear()
            st.session_state.app_terminated = True
            st.session_state.messages = []  # إعادة التهيئة فورًا بعد المسح
            return responses

        is_convo_handled, convo_responses = handle_general_conversation_query(query)
        if is_convo_handled:
            responses.extend(convo_responses)
            return responses

        # Only ask about web search if a new, non-empty response was generated
        if responses:  # Check if responses is not empty
            responses.extend(
                generate_assistant_response_audio(
                    f"I'm not sure how to handle '{query}'.", force_speak_st=True
                )
            )
            responses.extend(
                generate_assistant_response_audio(
                    "Shall I search the web for it?", force_speak_st=True
                )
            )
            st.session_state.last_assistant_question_context = {
                "type": "web_search_confirm",
                "query": query,
            }
        else:
            print(f"[Unknown Suppressed for ST] '{query}'. Music playing.")
            responses.append(
                {
                    "role": "assistant",
                    "text_content": f"(Didn't understand '{query}', and music is playing.)",
                    "audio_path": None,
                }
            )
    else:
        print("No intent detected.")
        responses.extend(
            generate_assistant_response_audio(
                "I'm sorry, I didn't understand. Can you please rephrase?", force_speak_st=True
            )
        )
    return responses


def get_nearest_reminder():
    """Gets the nearest upcoming reminder."""
    reminders = get_reminders_db()
    if not reminders:
        return None
    now = datetime.datetime.now()
    nearest_reminder = None
    nearest_time_diff = None
    for r_id, desc, dt_str, repeat in reminders:
        try:
            reminder_time = datetime.datetime.fromisoformat(dt_str)
            if reminder_time > now:
                time_diff = reminder_time - now
                if (
                    nearest_time_diff is None
                    or time_diff < nearest_time_diff
                ):
                    nearest_reminder = (r_id, desc, reminder_time.strftime('%A, %B %d at %I:%M %p'), repeat)
                    nearest_time_diff = time_diff
        except ValueError:
            print(f"Error parsing reminder datetime: {dt_str} for ID {r_id}. Skipping.")
            continue
    return nearest_reminder

# --- Online Image Generation ---

from io import BytesIO
from PIL import Image
import base64  # Added import


@st.cache_resource
def load_image_generation_model():
    """Placeholder for online service, no model to load"""
    return None, None  # No local model needed for online service

def generate_image(prompt, api_key="sk-uUSmmTwWvM4gucQMLLLwkXPnrE5LfZAvsLvUvraSso1nUFDE", service="stability"):  # IMPORTANT: set your key
    """Generates images using an online API"""
    try:
        if service == "stability":
            # Using Stability AI API (example)
            url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "text_prompts": [{"text": prompt, "weight": 1}],
                "cfg_scale": 7,
                "height": 1024,
                "width": 1024,
                "samples": 1,
                "steps": 30
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            data = response.json()
            if "artifacts" in data and data["artifacts"]:
                image_url = data["artifacts"][0]["base64"]
                image = Image.open(BytesIO(base64.b64decode(image_url)))
                return [image]
            else:
                st.error(f"Stability AI API error: {data.get('error', 'No artifacts found')}")
                return []
            
        elif service == "openai":
            try:
                import openai
                openai.api_key = "YOUR_OPENAI_API_KEY"  # IMPORTANT: Set your OpenAI API key
                response = openai.Image.create(
                    prompt=prompt,
                    n=1,
                    size="1024x1024"
                )
                image_url = response['data'][0]['url']
                image = Image.open(requests.get(image_url, stream=True).raw)
                return [image]
            except ImportError:
                st.error("OpenAI library not installed. Please install it with 'pip install openai'")
                return []
            except openai.error.OpenAIError as e:  # Correct exception type
                st.error(f"OpenAI API error: {e}")
                return []
            
        else:
            st.error("Unsupported online service")
            return []
            
    except requests.exceptions.RequestException as e:  # Correct Exception Handling
        st.error(f"Online image generation network error: {e}")
        return []
    except Exception as e:
        st.error(f"Online image generation error: {e}")
        return []


def run_streamlit_app_ui1():
    st.write("App started!") 
    global script_dir

    # --- Session state initializations ---
    if "messages" not in st.session_state or st.session_state.messages is None:
        st.session_state.messages = []
    if "generated_images" not in st.session_state:   # أضف هذا السطر
        st.session_state.generated_images = []
    if "pending_alerts_st" not in st.session_state:
        st.session_state.pending_alerts_st = []
    if "last_assistant_question_context" not in st.session_state:
        st.session_state.last_assistant_question_context = None
    if "is_music_playing_st" not in st.session_state:
        st.session_state.is_music_playing_st = False
    if "current_music_url_st" not in st.session_state:
        st.session_state.current_music_url_st = None
    if "current_music_title_st" not in st.session_state:
        st.session_state.current_music_title_st = None
    if "user_input_text" not in st.session_state:
        st.session_state.user_input_text = ""
    if "reminder_flow_step" not in st.session_state:
        st.session_state.reminder_flow_step = None
    if "reminder_data" not in st.session_state:
        st.session_state.reminder_data = {}
    if "app_terminated" not in st.session_state:
        st.session_state.app_terminated = False
    if "db_initialized" not in st.session_state:
        init_db()
        st.session_state.db_initialized = True
    if "threads_started" not in st.session_state:
        try:
            reminder_thread = threading.Thread(
                target=check_reminders, daemon=True, name="ReminderThread"
            )
            reminder_thread.start()
            st.session_state.threads_started = True
            print("Background threads started.")
        except Exception as e_thread:
            print(f"Failed to start background threads: {e_thread}")
            st.error(f"Failed to start background threads: {e_thread}")
            st.stop()
    if "email_login_needed" not in st.session_state:
        st.session_state.email_login_needed = False
    if "email_flow_step" not in st.session_state:
        st.session_state.email_flow_step = None # added this
    if "email_recipient" not in st.session_state:
        st.session_state.email_recipient = None # and this for more control of email
    if "user_email" not in st.session_state:  # Initialize email
        st.session_state.user_email = None
    if "user_password" not in st.session_state:
        st.session_state.user_password = None
    if "voice_input_processed" not in st.session_state: # Check if voice input was processed
        st.session_state.voice_input_processed = False 

    # --- Login check and form ---
    if not st.session_state.get("user_email") or not st.session_state.get("user_password") or st.session_state.email_login_needed:  # Show login if not logged in or email is specifically needed
        st.session_state.is_music_playing_st = False  # Stop music during login
        st.session_state.current_music_url_st = None
        st.session_state.current_music_title_st = None

        st.title("🔐 Login to VAIA")
        with st.form(key="login_form"):
            st.session_state.user_email = st.text_input("Email Address")  # Corrected
            st.session_state.user_password = st.text_input("Password", type="password")  # Corrected
            login_button = st.form_submit_button("Login")

        if login_button:
            if not st.session_state.user_email or not st.session_state.user_password:
                st.error("Please enter both email and password.")
            else: # Login success (basic)

                st.success("Login successful!")
                st.session_state.email_login_needed = False #Login is done
                st.rerun()  # Rerun after login to show main app
        st.stop() # Add stop here, so only Login is shown

    # --- Main layout ---
    with st.sidebar:  # Sidebar for controls and information
        logo_path = os.path.join(script_dir, "logo.jpeg")  # Path to the logo image
        if os.path.exists(logo_path):
            st.image(logo_path, width=200)  # Display the logo
        else:
            st.markdown(
                "<div style='text-align: center; font-size: 50px;'>🤖</div>",
                unsafe_allow_html=True,
            )  # If no logo, use an emoji
        st.title("VAIA Control Panel")
        st.markdown("---")

        # -- Speech input part --
        st.markdown("### 🎙️ Voice Input (Browser Microphone)")
        audio = audiorecorder(
            "Click to record", "Recording...", key="browser_mic"
        )  # Start recording button and stop when click again

        if len(audio) > 0:  # When recording is done
            st.success("✅ Recording done!")
            
            if not st.session_state.voice_input_processed:
                with NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ) as tmpfile:
                    audio.export(tmpfile, format="wav")
                    audio_path = tmpfile.name
                recognizer = sr.Recognizer()
                try:
                    with sr.AudioFile(audio_path) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data, language="en-US")
                        st.session_state.user_input_text = ""  # Clear input box

                        # Add user message
                        st.session_state.messages.append({"role": "user", "content": text, "source": "voice"})

                        with st.spinner("Thinking..."):
                            assistant_responses = process_command_logic_st(text)

                        # *** CRITICAL FIX: Handle assistant responses carefully ***
                        # use a list to maintain the message order,
                        # using a set for checking existing messages 
                        existing_messages = {(msg["role"], msg["content"]) for msg in st.session_state.messages}
                        new_messages_to_add = []

                        for res in assistant_responses:
                            message_key = ("assistant", res["text_content"])
                            if message_key not in existing_messages:
                                new_messages_to_add.append(
                                    {
                                        "role": "assistant",
                                        "content": res["text_content"],
                                        "audio_data": res["audio_path"],
                                    }
                                )
                                existing_messages.add(message_key)


                        st.session_state.messages.extend(new_messages_to_add) # Add all new messages at once
                        st.session_state.voice_input_processed = True 

                        try:
                            os.remove(audio_path)
                        except:
                            pass
                        
                        st.rerun()  # Rerun only *once* after processing all responses

                except sr.UnknownValueError:
                    st.error("❌ Could not understand the audio. Please try again.")
                except sr.RequestError as e:
                    st.error(f"❌ Could not request results from Google Speech Recognition service; {e}")
            
            # Reset flag, so it can be processed on next recording
            st.session_state.voice_input_processed = False  
        # -- Stop speech input part --
        st.markdown("---")
        if st.session_state.get("is_music_playing_st") and st.session_state.get(
            "current_music_title_st"
        ):
            st.subheader("🎵 Now Playing")
            st.caption(st.session_state.current_music_title_st)  # Show the song
            if st.button("⏹️ Stop Music", use_container_width=True):
                assistant_responses = stop_song_st()  # Play the song
                for res in assistant_responses:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": res["text_content"],
                            "audio_data": res["audio_path"],
                        }
                    )
                st.rerun()

        st.markdown("---")
        with st.expander("💡 Assistant Guide & Tips", expanded=False):  # Show some help
                st.markdown(
                    """
                **Welcome to VAIA - Your Voice & Text Assistant!**

                You can interact with VAIA using text or by clicking the "Record Voice Command" button (uses the server's microphone).

                **What VAIA Can Do:**
                *   🗣️ **General Chit-Chat:** Ask "How are you?", "Tell me a joke", etc.
                *   📧 **Send Email:** (Login required) "Send an email to [email address]"
                *   🎵 **Play Music:** "Play [song name or artist]" (e.g., "Play Imagine Dragons Believer")
                *   🛑 **Stop music":** "Stop music"
                *   🌦️ **Weather:** "What's the weather in [city]?"
                *   ⏰ **Reminders:**
                    *   "Set a reminder" / "Add reminder"
                    *   "View reminders" / "Show my reminders"
                    *   "Remove reminder" / "Delete reminder"
                    *   VAIA will guide you through the details. *Reminder alerts are currently server-side audio and UI notifications.*
                *   🌐 **Web Search & Summaries:**
                    *   "Summarize [topic]" (e.g., "Summarize artificial intelligence")
                    *   "Detailed search for [topic]"
                    *   "Google [search term]" (opens search on server)
                *   🈯 **Translate:** "Translate '[phrase]' to [language]" (e.g., "Translate 'hello world' to French")
                *   🖥️ **Server System Info:** "System status", "Computer health"
                *   🔊 **Server Volume:** "Increase volume", "Decrease volume", "Mute volume"
                *   📱 **Open on Server:** "Open Facebook", "Open YouTube"
                *   🛠️ **Open Apps on Server:** "Open notepad", "Open calculator" (Windows example)
                *   📸 **Server Screenshot:** "Take a screenshot"
                *   🌍 **Server Location:** "Where am I?"
                *   👋 **Exit:** "Goodbye", "Exit"

                **Important Notes:**
                *   Actions like opening applications, websites, changing volume, or taking screenshots happen on the **server** where this app is running, not your local computer.
                *   Voice recording uses the microphone connected to the **server**.
                *   Music and assistant speech will play through **your browser**.
                """
                )
                st.markdown("---")
                st.caption(f"VAIA v1.1 - Running in: {script_dir}")

    # Main Chat Interface
       # Main Chat Interface
    st.title("🤖 VAIA - Your Intelligent Assistant")
    st.markdown("Type your command below or use the voice input button in the sidebar.")

    # Display nearest upcoming reminder
    nearest_reminder = get_nearest_reminder()
    if nearest_reminder:
        r_id, desc, dt_str, repeat = nearest_reminder
        st.info(f"🔔 **Upcoming Reminder:** {desc} on {dt_str} (Repeats: {repeat})")
    else:
        st.info("✅ No reminders set.") # Inform user if no reminders are set

    # Display pending alerts
    if st.session_state.pending_alerts_st:
        alerts_to_show = st.session_state.pending_alerts_st[:]
        st.session_state.pending_alerts_st = []
        for alert in alerts_to_show:
            st.info(alert)

    # Show generated images if they exist before displaying chat messages
    if st.session_state.generated_images:
        st.subheader("Generated Images")  # Add a header above the images
        image_columns = st.columns(len(st.session_state.generated_images))
        for i, img_data in enumerate(st.session_state.generated_images):
            with image_columns[i]:
                st.image(img_data["image"], caption=img_data["caption"], use_column_width=True)  # Display the images
    
    # Display chat messages, filter duplicate user messages
    displayed_user_messages = set()
    for i, msg_data in enumerate(st.session_state.messages):
        if msg_data["role"] == "user":
            if (msg_data["content"]) not in displayed_user_messages: #Check for duplicate
                with st.chat_message("user", avatar="🧑‍💻"):  # Set user avatar
                    st.write(msg_data["content"])  # Show the content
                displayed_user_messages.add(msg_data["content"]) #Add to displayed set


        elif msg_data["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):  # Set assistant avatar
                st.write(msg_data["content"])  # Show the content
                if msg_data.get("audio_data"):
                    try:
                        with open(
                            msg_data["audio_data"], "rb"
                        ) as audio_file:  # Read audio bytes from file
                            audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/mp3")
                        if (
                            os.path.exists(msg_data["audio_data"])
                            and "temp" in msg_data["audio_data"].lower()
                            and msg_data["audio_data"].endswith(".mp3")
                        ):
                            try:
                                os.remove(
                                    msg_data["audio_data"]
                                )  # Remove the temporary audio
                            except Exception as e_rem:
                                print(
                                    f"Could not remove temp audio {msg_data['audio_data']}: {e_rem}"
                                )
                    except FileNotFoundError:
                        st.caption("(Audio file not found or already cleaned up)")
                    except Exception as e_audio_display:
                        st.caption(f"(Could not play audio: {e_audio_display})")

    
    if st.session_state.get("is_music_playing_st") and st.session_state.get(
        "current_music_url_st"
    ):
        st.audio(
            st.session_state.current_music_url_st, format="audio/mp3"
        )  # Display audio player

    # Input part of user messages
    with st.form(key="input_form", clear_on_submit=True):
        user_query_from_text = st.text_input(
            "Your command:",
            value=st.session_state.user_input_text,
            placeholder="Type here or use voice input...",
            label_visibility="collapsed",
        )
        submit_button = st.form_submit_button(label="➡️ Send")

    if submit_button and user_query_from_text:
        st.session_state.user_input_text = ""

        if st.session_state.app_terminated:
            st.info("Session was terminated. Please refresh to start a new one.")
            st.stop()

        st.session_state.messages.append(
            {"role": "user", "content": user_query_from_text, "source": "text"}
        )

        with st.spinner("Thinking..."):
            assistant_responses = process_command_logic_st(
                user_query_from_text
            )  # Message processing

        # *** CRITICAL FIX: Handle assistant responses carefully ***
        # use a list to maintain the message order,
        # using a set for checking existing messages 
        existing_messages = {(msg["role"], msg["content"]) for msg in st.session_state.messages}
        new_messages_to_add = []

        for res in assistant_responses:
            message_key = ("assistant", res["text_content"])
            if message_key not in existing_messages:
                new_messages_to_add.append(
                    {
                        "role": "assistant",
                        "content": res["text_content"],
                        "audio_data": res["audio_path"],
                    }
                )
                existing_messages.add(message_key)


        st.session_state.messages.extend(new_messages_to_add) # Add all new messages at once

        if st.session_state.app_terminated:
            st.info("Session has ended. Refresh to restart.")
            st.rerun()
        st.rerun()

    elif not st.session_state.messages:  # If it's the first message from the AI
        if "initial_greeting_done" not in st.session_state:
            initial_responses = wishMe_st()
            for res in initial_responses:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": res["text_content"],
                        "audio_data": res["audio_path"],
                    }
                )
            st.session_state.initial_greeting_done = True
            st.rerun()


# --- Load Model and Data (for AI intent recognition) ---
# This should run only once, Streamlit's @st.cache_resource can help

script_dir = os.path.dirname(os.path.abspath(__file__))

# Define SelfAttention Layer Here (Ensure same definition as training script)
class SelfAttention(tf.keras.layers.Layer):  # Ensure you inherit from tf.keras.layers.Layer
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.query = tf.keras.layers.Dense(units)
        self.key = tf.keras.layers.Dense(units)
        self.value = tf.keras.layers.Dense(units)

    def call(self, inputs):
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.units, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(weights, v)
        return output

    def get_config(self):  # VERY IMPORTANT: Implement get_config
        config = super().get_config()
        config.update({'units': self.units})  # Include units in config
        return config


@st.cache_resource
def load_ai_model_and_data():
    """Loads the AI model, tokenizer, label encoder, and intent data."""
    intents_path_local = os.path.join(script_dir, "intents[1].json")
    model_path_local = os.path.join(script_dir, "enhanced_chatbot_model.keras")
    tokenizer_path_local = os.path.join(script_dir, "enhanced_tokenizer.pkl")
    label_encoder_path_local = os.path.join(script_dir, "enhanced_label_encoder.pkl")

    import json, pickle # Import inside the function

    try:
        with open(intents_path_local, encoding="utf-8") as file:
            data_local = json.load(file)

        # THIS IS THE CRITICAL PART: Add the custom_objects argument
        model_local = load_model(
            model_path_local, 
            custom_objects={'SelfAttention': SelfAttention}  # Pass the custom layer
        )

        with open(tokenizer_path_local, "rb") as f:
            tokenizer_local = pickle.load(f)
        with open(label_encoder_path_local, "rb") as encoder_file:
            label_encoder_local = pickle.load(encoder_file)
        return data_local, model_local, tokenizer_local, label_encoder_local
    except FileNotFoundError as e:
        st.error(f"Error: Critical model/data file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or data files: {e}")
        st.stop()

# Load the AI model and data (once)
data, model, tokenizer, label_encoder = load_ai_model_and_data()

def check_reminders():
    """Checks for reminders and triggers alerts (server-side)."""
    print("Reminder checking thread started.")
    while True:
        try:
            now = datetime.datetime.now()
            reminders_to_check = get_reminders_db()
            for r_id, desc, dt_str, repeat in reminders_to_check:
                try:
                    reminder_time = datetime.datetime.fromisoformat(
                        dt_str
                    )
                except ValueError:
                    print(
                        f"Error parsing reminder datetime: {dt_str} for ID {r_id}. Skipping."
                    )
                    continue

                if now >= reminder_time:
                    alert_message = f"Reminder: {desc}"
                    print(
                        f"[Reminder Alert] ID {r_id}: {desc} at {dt_str}, Repeat: {repeat}"
                    )
                    speak_server_pyttsx3(alert_message)
                    # Add to a global list that Streamlit can periodically check
                    if "st" in globals() and "pending_alerts_st" in st.session_state:
                        # Show next reminder if no next reminder it wont add a next reminder
                        nearest_reminder = get_nearest_reminder()
                        if nearest_reminder:

                            r_id, desc, dt_str, repeat = nearest_reminder
                            st.session_state.pending_alerts_st.append(
                                f"🔔 Reminder: {desc} (Due: {dt_str})"
                            )  # This will show the next reminder
                        else:
                            st.session_state.pending_alerts_st.append(
                                f"🔔 Reminder: {desc} (Due: {dt_str})"
                            )  # If its over show the last reminder


                    conn = None
                    try:
                        conn = sqlite3.connect(
                            DATABASE_NAME, timeout=10, check_same_thread=False
                        )
                        c = conn.cursor()
                        if repeat.lower() == "once":
                            c.execute(
                                "DELETE FROM reminders WHERE id = ?", (r_id,)
                            )
                            print(f"Reminder ID {r_id} (once) removed.")
                        else:
                            next_reminder_time = reminder_time
                            valid_repeat = False
                            if repeat.lower() == "daily":
                                while next_reminder_time <= now:
                                    next_reminder_time += datetime.timedelta(
                                        days=1
                                    )
                                valid_repeat = True
                            elif repeat.lower() == "weekly":
                                while next_reminder_time <= now:
                                    next_reminder_time += datetime.timedelta(
                                        weeks=1
                                    )
                                valid_repeat = True

                            if valid_repeat:
                                c.execute(
                                    "UPDATE reminders SET datetime = ? WHERE id = ?",
                                    (next_reminder_time.isoformat(), r_id),
                                )
                                print(
                                    f"Reminder ID {r_id} ({repeat}) rescheduled to {next_reminder_time.isoformat()}."
                                )
                            else:
                                c.execute(
                                    "DELETE FROM reminders WHERE id = ?", (r_id,)
                                )  # Delete if repeat type is unknown
                                print(
                                    f"Reminder ID {r_id} (unknown repeat '{repeat}') removed."
                                )
                        conn.commit()
                    except sqlite3.Error as e_sql_update:
                        print(
                            f"DB Error updating/deleting reminder ID {r_id}: {e_sql_update}"
                        )
                    finally:
                        if conn:
                            conn.close()
            time.sleep(30)  # Check every 30 seconds
        except sqlite3.Error as e_sql_main:
            print(f"SQLite error in check_reminders: {e_sql_main}")
            time.sleep(60)
        except Exception as e:
            print(f"Unexpected error in check_reminders: {e}")
            import traceback

            traceback.print_exc()
            time.sleep(60)

# --- Main Execution ---
if __name__ == "__main__":
    if (
        OPENWEATHERMAP_API_KEY == "YOUR_ACTUAL_OPENWEATHERMAP_API_KEY"
        or not OPENWEATHERMAP_API_KEY
        or len(OPENWEATHERMAP_API_KEY) < 30
    ):
        print(
            "\n"
            + "=" * 60
            + "\nWARNING: OPENWEATHERMAP_API_KEY IS NOT SET or is a placeholder!\nWeather functionality WILL FAIL.\nPlease get a key from openweathermap.org and update the script.\n"
            + "=" * 60
            + "\n"
        )

    initialize_pyttsx3_engine()

    # Initialize database and background threads
    if "db_initialized" not in st.session_state:
        init_db()
        st.session_state.db_initialized = True
    if "threads_started" not in st.session_state:
        try:
            reminder_thread = threading.Thread(
                target=check_reminders, daemon=True, name="ReminderThread"
            )
            reminder_thread.start()
            st.session_state.threads_started = True
            print("Background threads started.")
        except Exception as e_thread:
            print(f"Failed to start background threads: {e_thread}")
            st.error(f"Failed to start background threads: {e_thread}")
            st.stop()

    if hasattr(sys, "argv") and any("streamlit" in arg for arg in sys.argv):
        run_streamlit_app_ui1()
    else:
        print(
            "Running in command-line mode (not Streamlit). CLI mode is not fully featured in this version."
        )
        pass
