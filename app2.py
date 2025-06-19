import pyttsx3
import threading
import time
import sys
import os
import datetime
import webbrowser
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
import yt_dlp
import requests
import re
from deep_translator import GoogleTranslator
import sqlite3
from gtts import gTTS
import tempfile
import base64  # Added import
import pyautogui

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


# --- Global Variables ---
OPENWEATHERMAP_API_KEY = "661f67d8cc9df0c9e161f20f23eb60d7"  # Replace with your actual API key
player = None  # VLC player instance (not used in this version)
opened_processes = {}  # For tracking processes
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
DATABASE_NAME = os.path.join(script_dir, "assistant_data.db")  # Database file path


# --- Helper Functions ---
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
        print("Database initialized successfully.")
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()


def initialize_pyttsx3_engine():
    """Initializes the pyttsx3 text-to-speech engine."""
    global tts_engine
    if tts_engine is None:
        with tts_engine_lock:
            if tts_engine is None:
                try:
                    engine = pyttsx3.init()  # Platform auto-detection
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
                    engine.setProperty("rate", max(50, rate - 70))  # Adjust speaking rate
                    volume = engine.getProperty("volume")
                    engine.setProperty("volume", min(1.0, volume + 0.25))  # Adjust volume
                    tts_engine = engine
                except Exception as e:
                    print(f"Error initializing pyttsx3 engine: {e}")
                    tts_engine = None
    return tts_engine


def speak(text, wait=False):
    """Speaks text using pyttsx3."""
    global tts_engine_lock, tts_engine
    engine_instance = initialize_pyttsx3_engine()
    if engine_instance:
        with tts_engine_lock:
            try:
                engine_instance.say(text)
                engine_instance.runAndWait()
            except RuntimeError:
                print("TTS Runtime Error. Reinitializing engine.")
                tts_engine = None
                engine_instance_retry = initialize_pyttsx3_engine()
                if engine_instance_retry:
                    try:
                        engine_instance_retry.say(text)
                        engine_instance_retry.runAndWait()
                    except Exception as e_retry:
                        print(f"TTS Error on retry: {e_retry}")
                else:
                    print("TTS reinitialization failed.")
            except Exception as e:
                print(f"Error during speech: {e}")
        if wait:
            time.sleep(0.5)
    else:
        print(f"TTS SKIPPED (engine not initialized): {text}")


def speak_random(phrases):
    """Speaks a random phrase."""
    speak(random.choice(phrases))


def command(recognizer_instance=None, prompt_text=None):
    """Listens for audio input using the microphone and transcribes it."""
    current_recognizer = (
        recognizer_instance if recognizer_instance else main_recognizer
    )
    if prompt_text:
        print(f"Prompting: {prompt_text}")
        speak(prompt_text)

    try:
        with sr.Microphone() as source:
            try:
                current_recognizer.adjust_for_ambient_noise(source, duration=0.5)
            except Exception as e_adjust:
                print(f"Could not adjust for ambient noise: {e_adjust}")

            print("Listening...", end="", flush=True)
            current_recognizer.pause_threshold = 0.8
            audio = current_recognizer.listen(source, timeout=7, phrase_time_limit=12)

            print("\rRecognizing...", end="", flush=True)
            query = current_recognizer.recognize_google(audio, language="en-US")
            print(f"\rUser said: {query}\n")
            return query.lower()
    except sr.WaitTimeoutError:
        print("\rNo speech detected.")
        speak("I didn't hear anything.")
        return None
    except sr.UnknownValueError:
        print("\rCould not understand audio.")
        speak("Sorry, I couldn't understand that.")
        return "unintelligible"
    except sr.RequestError as e:
        print(f"\rSpeech service request error; {e}")
        speak_random(
            [
                "My speech recognition is offline.",
                "Connection issue with speech service.",
            ]
        )
        return None
    except OSError as e_os:
        print(f"\rOSError with microphone: {e_os}. Check microphone.")
        speak("Microphone access error. Please check its connection.")
        return None
    except Exception as e:
        print(f"\rUnexpected error in command(): {e}")
        import traceback

        traceback.print_exc()
        speak("An unexpected error occurred with voice input.")
        return None


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
        print(f"Reminder added: {desc} on {dt_string}, repeats {repeat_type}")
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


# --- Reminder Management Flow ---
def manage_reminders_flow():
    """Manages the reminder creation/viewing/removal flow in the CLI."""
    step = "action_prompted"
    reminder_data = {}

    while step:
        if step == "action_prompted":
            action = command(prompt_text="What would you like to do with reminders? Add, view, or remove?")
            if not action or action == "unintelligible":
                speak("I didn't catch that. Please say add, view, or remove.")
            elif "add" in action:
                step = "add_desc"
            elif "view" in action or "show" in action or "list" in action:
                step = "view"
            elif "remove" in action or "delete" in action:
                step = "remove_prompt"
            else:
                speak("Sorry, I didn't understand that. Try 'add', 'view', or 'remove'.")
                step = None

        elif step == "add_desc":
            desc = command(prompt_text="What is the reminder about?")
            if not desc:
                speak("Description cannot be empty. Cancelling reminder addition.")
                step = None
            else:
                reminder_data["desc"] = desc
                step = "add_date"

        elif step == "add_date":
            date_str = command(prompt_text="What date? (e.g., 'tomorrow', 'next Friday', 'July 25th', or YYYY-MM-DD)")
            if not date_str:
                speak("Date not provided. Cancelling.")
                step = None
                continue

            parsed_date_obj = None
            today = datetime.date.today()
            date_input_lower = date_str.lower()
            if "today" in date_input_lower:
                parsed_date_obj = today
            elif "tomorrow" in date_input_lower:
                parsed_date_obj = today + datetime.timedelta(days=1)
            elif "next" in date_input_lower:
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
                        date_str, "%Y-%m-%d"
                    ).date()  # YYYY-MM-DD format
                except ValueError:
                    pass
            if not parsed_date_obj:
                try:  # Month Day Year format
                    match_mdy = re.search(
                        r"(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?",
                        date_str,
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
            if not parsed_date_obj:
                speak(
                    "Could not understand date. Try 'Month Day Year' or 'YYYY-MM-DD', 'tomorrow', etc. Please provide the date again."
                )
            else:
                reminder_data["date_str"] = parsed_date_obj.isoformat()
                step = "add_time"

        elif step == "add_time":
            time_str = command(prompt_text="What time? (e.g., 9 AM, 2:30 PM, 14:00)")
            if not time_str:
                speak("Time not provided. Cancelling.")
                step = None
                continue

            time_obj = None
            time_input_raw = time_str
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

            if not time_obj:
                speak(
                    "Invalid time. Try HH:MM, H AM/PM (e.g. 9 AM, 2:30 PM, 14:00). Please provide the time again."
                )
            else:
                reminder_data["time_str_iso"] = time_obj.strftime("%H:%M:%S")
                step = "add_repeat"

        elif step == "add_repeat":
            repeat_str = command(prompt_text="Repeat? (once, daily, weekly, or none)")
            repeat_type = "once"
            if repeat_str:
                repeat_input_lower = repeat_str.lower()
                if "daily" in repeat_input_lower:
                    repeat_type = "daily"
                elif "weekly" in repeat_input_lower:
                    repeat_type = "weekly"
            reminder_data["repeat_type"] = repeat_type

            try:
                final_dt_obj = datetime.datetime.fromisoformat(
                    f"{reminder_data['date_str']}T{reminder_data['time_str_iso']}"
                )
                if (
                    final_dt_obj <= datetime.datetime.now() + datetime.timedelta(minutes=1)
                ):
                    speak("The reminder time is in the past or too soon. Please start over.")
                else:
                    add_reminder_db(
                        reminder_data["desc"],
                        final_dt_obj.isoformat(),
                        reminder_data["repeat_type"],
                    )
                    speak(
                        f"Okay, I've set a {reminder_data['repeat_type']} reminder for {reminder_data['desc']} on {final_dt_obj.strftime('%A, %B %d at %I:%M %p')}."
                    )
            except Exception as e:
                print(f"Error setting reminder: {e}")
                speak(f"Sorry, there was an error setting the reminder: {e}")

            step = None  # End of flow

        elif step == "view":
            reminders = get_reminders_db()
            if not reminders:
                speak("You have no reminders set.")
            else:
                speak("Here are your reminders:")
                for r_id, desc, dt_str, repeat in reminders:
                    try:
                        dt_obj = datetime.datetime.fromisoformat(dt_str)
                        speak(
                            f"Number {r_id}: {desc}, on {dt_obj.strftime('%A, %B %d at %I:%M %p')}. Repeats: {repeat}."
                        )
                    except ValueError:
                        speak(
                            f"Number {r_id}: {desc}, at invalid time '{dt_str}'. Repeats: {repeat}."
                        )
            step = None  # End of flow

        elif step == "remove_prompt":
            reminders = get_reminders_db()
            if not reminders:
                speak("You have no reminders to remove.")
                step = None
                continue

            print("Reminders:")
            for r_id, desc, dt_str, repeat in reminders:
                try:
                    dt_obj = datetime.datetime.fromisoformat(dt_str)
                    print(
                        f"{r_id}: {desc} ({dt_obj.strftime('%b %d, %I:%M%p')})"
                    )
                except ValueError:
                    print(f"{r_id}: {desc} (invalid date)")

            id_to_remove_str = command(prompt_text="Which one to remove? Say the number or part of the description.")
            if not id_to_remove_str or id_to_remove_str == "unintelligible":
                speak("No selection made for removal. Cancelling.")
                step = None
                continue

            removed_flag = False
            if id_to_remove_str.isdigit():
                target_id = int(id_to_remove_str)
                if any(r[0] == target_id for r in reminders):
                    if remove_reminder_db(target_id):
                        speak(f"Reminder number {target_id} removed.")
                        removed_flag = True
                    else:
                        speak(f"Error removing reminder number {target_id}.")
                else:
                    speak(f"No reminder found with number {target_id}.")

            if not removed_flag and not id_to_remove_str.isdigit():
                matched_reminders = [
                    r
                    for r_id_match, desc_db, _, _ in reminders
                    if id_to_remove_str in desc_db.lower()
                    for r in [(r_id_match, desc_db)]
                ]
                if not matched_reminders:
                    speak(
                        f"Could not find a reminder matching '{id_to_remove_str}'. Try the number."
                    )
                elif len(matched_reminders) == 1:
                    r_id_match, desc_match = matched_reminders[0]
                    step = "remove_confirm"
                    reminder_data["id_to_confirm"] = r_id_match
                    reminder_data["desc_to_confirm"] = desc_match
                    speak(
                        f"Found: Number {r_id_match}, '{desc_match}'. Remove it? (yes/no)"
                    )
                else:
                    speak(
                        "Multiple reminders match that description. Please specify by number:"
                    )
                    for r_id_multi, desc_multi in matched_reminders:
                        speak(f"Number {r_id_multi}: {desc_multi}")

            if removed_flag or not matched_reminders and not id_to_remove_str.isdigit():
                step = None  # End flow

        elif step == "remove_confirm":
            confirm_action = command(prompt_text="Confirm removal? (yes/no)")
            if confirm_action and confirm_action in ["yes", "yeah", "ok", "yep"]:
                if remove_reminder_db(reminder_data["id_to_confirm"]):
                    speak(f"Reminder '{reminder_data['desc_to_confirm']}' removed.")
                else:
                    speak(f"Error removing reminder '{reminder_data['desc_to_confirm']}'.")
            else:
                speak(f"Okay, reminder '{reminder_data['desc_to_confirm']}' was not removed.")
            step = None  # End flow


def get_weather_info(city_name):
    """Gets and speaks weather information for a city."""
    if (
        OPENWEATHERMAP_API_KEY == "YOUR_ACTUAL_OPENWEATHERMAP_API_KEY"
        or not OPENWEATHERMAP_API_KEY
        or len(OPENWEATHERMAP_API_KEY) < 30
    ):
        msg = "Weather service is not configured correctly. Please add a valid OpenWeatherMap API key."
        print(msg)
        speak(msg)
        return

    params = {"q": city_name.strip(), "appid": OPENWEATHERMAP_API_KEY, "units": "metric"}
    try:
        response = requests.get(
            "http://api.openweathermap.org/data/2.5/weather?", params=params, timeout=10
        )
        response.raise_for_status()
        weather_data = response.json()
        if str(weather_data.get("cod")) == "404":
            speak(f"Sorry, {weather_data.get('message', 'city not found')}.")
        elif weather_data.get("main") and weather_data.get("weather"):
            main, weather = weather_data["main"], weather_data["weather"][0]
            wind_spd = f"{weather_data.get('wind', {}).get('speed', 0) * 3.6:.1f} km/h"
            report = f"In {weather_data.get('name', city_name.capitalize())}: Temperature is {main.get('temp')}° Celsius, feels like {main.get('feels_like')}° Celsius. The sky is {weather.get('description')}. Humidity is {main.get('humidity')} percent. Wind speed is {wind_spd}."
            speak(report)
        else:
            speak(f"I received an unexpected weather response for {city_name}.")
    except requests.exceptions.HTTPError as http_err:
        print(f"Weather HTTP error: {http_err}")
        if http_err.response.status_code == 401:  # Invalid API key
            speak("The weather API key is invalid or not authorized.")
        else:
            speak(
                f"There's a weather service error (code {http_err.response.status_code})."
            )
    except requests.exceptions.RequestException as e:
        print(f"Weather request error: {e}")
        speak("I can't connect to the weather service right now.")
    except Exception as e:
        print(f"Weather processing error: {e}")
        speak("Sorry, an error occurred while fetching weather information.")


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
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
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
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
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


def play_song(song_query):
    """Plays a song from YouTube in the CLI."""
    speak_random([f"Looking for {song_query}...", f"Finding {song_query}..."])
    video_url, video_title = search_youtube(song_query)
    if not video_url:
        speak(f"Sorry, I couldn't find '{song_query}' on YouTube.")
        return

    speak(f"Found '{video_title}'. Getting the audio stream...")
    playurl = get_best_audio_url(video_url)
    if not playurl:
        speak(
            f"I found '{video_title}', but I'm having trouble getting a playable audio stream for it."
        )
        return

    try:
        print(f"Attempting to play URL: {playurl}")
        speak(f"Now playing: {video_title}.")
        subprocess.Popen(["mpg123", playurl]) # Using mpg123 for command line music playback
    except Exception as e:
        print(f"Error playing audio: {e}")
        speak(f"Sorry, I encountered an error while trying to play the song.")


def stop_song():
    """Stops the currently playing song (CLI version)."""
    try:
        subprocess.run(["killall", "mpg123"], check=False)  # Kill mpg123 process
        speak_random(["Music stopped.", "Playback halted."])
    except Exception as e:
        print(f"Error stopping playback: {e}")
        speak("Error stopping music playback.")


def perform_web_search_and_summarize(query_text):
    """Performs a web search and summarizes the first result."""
    speak_random(
        [f"Looking up '{query_text}'...", f"Searching for '{query_text}'..."]
    )
    search_results_list = []
    try:
        try:
            print(f"Attempting web search for: {query_text}")
            search_results_iter = search(
                query_text, num_results=1, lang="en", sleep_interval=1
            )
            search_results_list = list(search_results_iter)
            print(f"Search results found: {search_results_list}")
        except Exception as e_search:
            print(f"Search Error: {e_search}")
            speak(f"I encountered a web search issue: {e_search}")
            return

        if not search_results_list:
            speak(f"I couldn't find any web results for '{query_text}'.")
            return

        first_url = next(
            (url for url in search_results_list if url.startswith("http")), None
        )
        if not first_url:
            speak("Sorry, I couldn't find a valid web page to summarize.")
            return
        print(f"Summarizing: {first_url}")
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}  # Add User-Agent
            print(f"Attempting to get content from: {first_url}")  # Log content fetch
            response_req = requests.get(
                first_url, headers=headers, timeout=15, verify=False, allow_redirects=True  # Disable SSL verification (temporary) and allow redirects
            )  # Get the page content
            response_req.raise_for_status()
            print(f"Content fetched successfully. Status code: {response_req.status_code}")  # Log status
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
                        raise  # Re-raise the exception if all parsing attempts fail
            
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
                    else "Could not extract meaningful summary."
                )  # Create summary
            else:
                page_content_summary = "No suitable paragraphs found for summary."
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
        and not page_content_summary.startswith("I had an issue") and not page_content_summary.startswith("SSL Verification Failed") and not page_content_summary.startswith("Request failed") and not page_content_summary.startswith("A major error occurred")
        else page_content_summary
    )
    speak(final_summary_text)
    if final_summary_text and not final_summary_text.startswith("SSL Verification Failed") and not final_summary_text.startswith("Request failed") and not final_summary_text.startswith("A major error occurred"):
        open_page_confirm = command(prompt_text="Would you like me to open this page for more details?")
        if open_page_confirm and any(
            word in open_page_confirm for word in ["yes", "sure", "okay", "do it"]
        ):
            open_webpage(first_url)


def web_search_detailed(query, num_results=1):
    """Performs a detailed web search and returns the results."""
    try:
        results = list(
            search(query, num_results=num_results, lang="en", sleep_interval=1)
        )
        results = [url for url in results if url.startswith("http")]
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
            )
            response.raise_for_status()
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
                el.decompose()
            texts = [
                el.get_text(" ", strip=True)
                for tag_name in ["article", "main", "section", "div", "p"]
                for el in soup.find_all(tag_name)
                if el.get_text(strip=True) and len(el.get_text(strip=True).split()) > 10
            ]
            if texts:
                content = ". ".join(texts)
                sentences = [
                    s.strip() for s in content.split(".") if s.strip()
                ]
                keywords = [
                    "study",
                    "research",
                    "found",
                    "show",
                    "reveal",
                    "important",
                    "key",
                    "conclusion",
                ]
                imp_sentences = sentences[:2]
                for s_item in sentences[2:8]:
                    if any(k in s_item.lower() for k in keywords) and len(
                        imp_sentences
                    ) < 5:
                        imp_sentences.append(s_item)
                return (
                    ". ".join(imp_sentences[:5]) + "."
                ).strip() if imp_sentences else "No specific important sentences found."
            return "No suitable content for detailed summary."
        except requests.exceptions.RequestException as e:
            print(f"Error getting detailed content for {url}: {e}")
            return f"Could not fetch or process {url} due to a request error: {e}"
        except Exception as e:
            print(f"Error getting detailed content for {url}: {e}")
            return f"An error occurred while fetching or processing: {url}. Error: {e}"
    except Exception as e:
        print(f"Outer Error getting detailed content for {url}: {e}")
        return f"A major error occurred while fetching or processing: {url}. Error: {e}"

def handle_detailed_web_search(query):
    """Handles detailed web searches and provides a summary."""
    speak(f"Okay, performing a detailed search for {query}...")
    results = web_search_detailed(query, num_results=1)
    if not results:
        speak("I didn't find any results for that detailed search.")
        return

    top_url = results[0]
    print(f"\nDetailed search top URL: {top_url}")
    speak("I found a page. Extracting detailed content now...")
    content_summary = get_page_content_detailed(top_url)

    if content_summary and not content_summary.startswith(
        "Could not"
    ) and not content_summary.startswith("An error"):
        print(f"\nDetailed Summary: {content_summary}\n")
        speak(f"Here's the detailed information I found: {content_summary[:300]}...")
        open_page_confirm = command(prompt_text="Should I open the page for all the details?")
        if open_page_confirm and any(
            word in open_page_confirm for word in ["yes", "sure", "okay", "do it"]
        ):
            open_webpage(top_url)
    else:
        print(f"Detailed content extraction failed: {content_summary}")
        speak(f"Sorry, I {content_summary.lower() if content_summary else 'could not extract detailed content.'}")


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
    is_handled = False

    if not query_text or query_text.strip() == "":
        print("Empty query provided to handle_general_conversation_query.")
        return is_handled

    # التنبؤ بالنوايا
    tag, confidence = predict_intent_from_text(query_text)
    
    if tag:
        print(f"Intent detected: {tag}, Confidence: {confidence:.3f}")

        # تجنب المعالجة إن كانت الثقة منخفضة
        if confidence < 0.65:
            print("Confidence too low for intent handling.")
            return is_handled

        # البحث عن النية المطابقة في intents.json
        for intent_data in data.get("intents", []):
            if intent_data["tag"].lower() == tag.lower():
                response_text = random.choice(intent_data["responses"])
                speak(response_text)  # Speak
                is_handled = True
                break
    else:
        print("No intent detected.")

    return is_handled

# --- System Interaction Functions ---
def wishMe():
    """Greets the user with a personalized message based on the time of day."""
    speak(
        f"Good {('Morning' if 0 <= datetime.datetime.now().hour < 12 else 'Afternoon' if 12 <= datetime.datetime.now().hour < 18 else 'Evening')}! How can I help?"
    )


def get_location():
    """Gets the location of the server based on its IP address."""
    try:
        data_loc = requests.get("https://ipinfo.io/json", timeout=7).json()
        speak(
            f"Based on the server's IP, it looks like it's near {data_loc.get('city', 'an unknown city')}, {data_loc.get('region', 'an unknown region')}."
        )
    except Exception as e:
        print(f"Location error: {e}")
        speak("I couldn't determine the server's location.")


def translate_text(text_to_translate, target_language_name):
    """Translates text to the specified language."""
    if not text_to_translate:
        speak("What text should I translate?")
        return

    target_lang_lower = target_language_name.lower().strip()
    if target_lang_lower not in language_map:
        supported_langs = ", ".join(list(language_map.keys())[:5])
        speak(
            f"I can't translate to {target_language_name}. I support languages like: {supported_langs}..."
        )
        return

    target_lang_code = language_map[target_lang_lower]
    translated_text, used_translator = None, "N/A"
    try:
        translated_text = GoogleTranslator(
            source="auto", target=target_lang_code
        ).translate(
            text_to_translate
        )
        used_translator = "DeepTranslator"
    except Exception as e_deep:
        print(f"DeepTranslator error: {e_deep}. Trying fallback.")
        if google_translate_service_instance:
            try:
                trans_obj = google_translate_service_instance.translate(
                    text_to_translate, dest=target_lang_code
                )
                translated_text = trans_obj.text
                used_translator = "googletrans"
            except Exception as e_trans:
                print(f"googletrans error: {e_trans}")

    if not translated_text:
        speak(f"I had some trouble translating to {target_language_name}.")
        return

    speak(
        f"In {target_language_name.capitalize()}, '{text_to_translate}' is: '{translated_text}'. (Translated using {used_translator})."
    )
    if target_lang_code != "en":
        say_translation_confirm = command(
            prompt_text=f"Would you like me to say that in {target_language_name.capitalize()}?"
        )
        if say_translation_confirm and any(
            word in say_translation_confirm for word in ["yes", "sure", "okay", "do it"]
        ):
            speak(translated_text, wait=True)


def open_social_media(platform_name):
    """Opens a social media platform in the default web browser."""
    urls = {
        "facebook": "https://facebook.com",
        "twitter": "https://twitter.com",
        "youtube": "https://youtube.com",
        "instagram": "https://instagram.com",
        "linkedin": "https://linkedin.com",
    }
    url = urls.get(platform_name.lower().strip())
    if url:
        speak(f"Opening {platform_name.capitalize()}.")
        webbrowser.open(url)
    else:
        speak(f"I don't have a link for {platform_name}.")


def open_application(app_name_query):
    """Opens an application on the system."""
    global opened_processes
    app_map = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "chrome": "chrome.exe",
    }  # Example for Windows
    cmd, name_app, exe_app = None, None, None
    query_lower = app_name_query.lower().strip()
    for n_app, c_app in app_map.items():
        if n_app in query_lower:
            cmd, name_app, exe_app = c_app, n_app.capitalize(), c_app
            break
    if cmd:
        speak(f"Attempting to open {name_app}.")
        try:
            proc = subprocess.Popen(cmd)
            opened_processes[exe_app.lower()] = proc
            speak(f"{name_app} should be opening.")
        except Exception as e:
            speak(f"Error opening {name_app}: {e}")
    else:
        speak(f"I don't know how to open '{app_name_query}'.")


def close_application(app_name_query):
    """Closes an application on the system."""
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
    if sys.platform == "win32" and exe_to_close:
        try:
            res = subprocess.run(
                ["taskkill", "/F", "/IM", exe_to_close],
                check=True,
                capture_output=True,
                text=True,
            )
            if "SUCCESS" in res.stdout:
                closed_msg = f"Closed {app_name_query}."
                if exe_to_close.lower() in opened_processes:
                    opened_processes.pop(exe_to_close.lower())
            else:
                closed_msg = f"Could not find or close {app_name_query}. Taskkill output: {res.stderr or res.stdout}"
        except Exception as e:
            closed_msg = f"Error closing {app_name_query}: {e}"
    elif exe_to_close.lower() in opened_processes:
        proc = opened_processes.pop(exe_to_close.lower())
        proc.terminate()
        try:
            proc.wait(timeout=2)
            closed_msg = f"Closed {app_name_query}."
        except subprocess.TimeoutExpired:
            proc.kill()
            closed_msg = f"Force closed {app_name_query}."
        except Exception as e:
            closed_msg = f"Error terminating {app_name_query}: {e}"
    else:
        closed_msg = f"I don't know how to close '{app_name_query}', or it wasn't opened by me."
    speak(closed_msg)


def perform_browsing(search_term):
    """Performs a web search in the default web browser."""
    try:
        webbrowser.open(
            f"https://www.google.com/search?q={requests.utils.quote(search_term.strip())}"
        )
        speak(f"Searching Google for '{search_term}'.")
    except Exception as e:
        speak(f"Error opening search: {e}")


def open_webpage(url):
    """Opens a webpage in the default web browser."""
    try:
        webbrowser.open(url)
        speak(f"Opening webpage: {url}.")
    except Exception as e:
        speak(f"Error opening webpage: {e}.")


def check_system_condition():
    """Checks the system condition (CPU, memory, disk usage)."""
    try:
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk_path = "/"
        if sys.platform == "win32":
            system_root = os.environ.get("SystemRoot", "C:\\")
            disk_path = os.path.splitdrive(system_root)[0] + os.sep

        disk_report_str = "Could not retrieve disk usage information."
        try:
            disk = psutil.disk_usage(disk_path)
            disk_report_str = (
                f"Disk space ({disk_path}) is {disk.percent}% used "
                f"({disk.free / 1024 ** 3:.1f}GB free)."
            )
        except Exception as e_disk:
            print(f"Error getting disk usage for '{disk_path}': {e_disk}")

        report_parts = [
            f"CPU is at {cpu}%",
            f"Memory is at {mem.percent}% used ({mem.available / 1024 ** 3:.1f}GB free)",
            disk_report_str,
        ]

        if hasattr(psutil, "sensors_battery"):
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
                report_parts.append(f"Battery is at {battery.percent}% ({plugged})")
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
                        f"About {int(hours)} hours and {int(minutes)} minutes remaining on battery"
                    )

        speak("System Status: " + ". ".join(report_parts) + ".")

    except Exception as e_psutil:
        print(f"Error getting system condition with psutil: {e_psutil}")
        speak("I'm having trouble getting the full system status at the moment.")


def change_system_volume(direction, amount=10):
    """Changes the system volume."""
    msg = ""
    try:
        if sys.platform == "win32":
            if direction == "up":
                pyautogui.press(
                    "volumeup", presses=amount // 5 if amount > 5 else 1
                )
            elif direction == "down":
                pyautogui.press(
                    "volumedown", presses=amount // 5 if amount > 5 else 1
                )
            elif direction in ["mute", "unmute"]:
                pyautogui.press("volumemute")
            msg = f"Volume {direction}."
        elif sys.platform == "darwin":
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
            msg = f"Volume {direction}."
        elif sys.platform.startswith("linux"):
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
                msg = "Volume muted."
            elif direction == "unmute":
                subprocess.run(["amixer", "-q", "sset", "Master", "unmute"], check=True)
                msg = "Volume unmuted."

            if direction in ["up", "down"]:
                subprocess.run(
                    ["amixer", "-q", "sset", "Master", f"{new_vol}%"],
                    check=True,
                )
                msg = f"Volume set to {new_vol} percent."
        else:
            msg = "Volume control not supported on this OS yet."
    except Exception as e:
        print(f"Error changing volume: {e}")
        msg = f"Could not change system volume: {e}"

    if msg:
        speak(msg)


def take_screenshot():
    """Takes a screenshot and saves it."""
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ss_dir = os.path.join(script_dir, "screenshots")
        os.makedirs(ss_dir, exist_ok=True)
        fn = os.path.join(ss_dir, f"ss_{ts}.png")
        pyautogui.screenshot(fn)
        speak(f"Screenshot saved in '{ss_dir}'.")
    except Exception as e:
        print(f"Screenshot error: {e}")
        speak("Error taking screenshot.")

# --- Intent Prediction and Loading Model ---
def load_ai_model_and_data():
    """Loads the AI model, tokenizer, label encoder, and intent data."""
    intents_path_local = os.path.join(script_dir, "intents.json")
    model_path_local = os.path.join(script_dir, "chat_model.keras")
    tokenizer_path_local = os.path.join(script_dir, "tokenizer.pkl")
    label_encoder_path_local = os.path.join(script_dir, "label_encoder.pkl")

    try:
        with open(intents_path_local, encoding="utf-8") as file:
            data_local = json.load(file)  # Load intent data (JSON)

        model_local = load_model(model_path_local)  # Load the Keras model
        with open(tokenizer_path_local, "rb") as f:
            tokenizer_local = pickle.load(f)  # Load the tokenizer
        with open(label_encoder_path_local, "rb") as encoder_file:
            label_encoder_local = pickle.load(encoder_file)  # Load the label encoder
        return data_local, model_local, tokenizer_local, label_encoder_local
    except FileNotFoundError as e:
        print(
            f"Error: Critical model/data file not found: {e}. Ensure files are in '{script_dir}'."
        )
        sys.exit(1)  # Stop execution if file is missing
    except Exception as e:
        print(f"Error loading model or data files: {e}")
        sys.exit(1)  # Stop execution if another error occurs

# Load the AI model and data (once)
data, model, tokenizer, label_encoder = load_ai_model_and_data()

# --- Main Processing Function ---
def process_command(query):
    """Processes user commands."""
    if not query:
        return

    lc_query = query.lower().strip()
    if "reminder" in lc_query:
        manage_reminders_flow()
        return
    
    if play_match := re.search(
        r"(?:play|listen to|put on|stream)\s+(?:the\s+)?(?:song\s+|music\s+|track\s+)?(.+)",
        query,
        re.IGNORECASE,
    ):
        song_name = play_match.group(1).strip()
        if song_name:
            play_song(song_name)
        else:
            speak("Sure, what song or artist would you like to hear?")
        return
    
    if re.search(
        r"stop\s*(?:the\s*)?(song|music|playback|playing)|stop current track",
        query,
        re.IGNORECASE,
    ):
        stop_song()
        return


    if m := re.search(
        r"(?:weather|forecast)(?:\s+(?:in|for|at|like in)\s+([a-zA-Z\s\-,'.]+))?",
        query,
        re.IGNORECASE,
    ):
        city = m.group(1).strip() if m.group(1) else None
        if city:
            get_weather_info(city)
        else:
            city_name = command(prompt_text="For which city would you like the weather forecast?")
            if city_name:
                get_weather_info(city_name)
        return

    elif re.search(r"where\s+am\s+i|my\s+location", query, re.IGNORECASE):
        get_location()
        return
    elif re.search(
        r"system\s+(status|condition|info)|computer health", query, re.IGNORECASE
    ):
        check_system_condition()
        return
    elif m := re.search(
        r"translate\s*(?:this|the phrase)?\s*['\"]?(.*?)['\"]?\s+to\s+([a-zA-Z\s\-]+)",
        query,
        re.IGNORECASE,
    ):
        translate_text(m.group(1).strip(), m.group(2).strip())
        return
    elif m := re.search(r"open\s+(.+)", query, re.IGNORECASE):
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
                open_social_media(s_key)
                opened_flag = True
                break
        if not opened_flag:
            app_to_open = re.sub(
                r"^(app|program|application)\s+", "", target, flags=re.I
            ).strip()
            open_application(app_to_open)
        return
    elif m := re.search(r"close\s+(.+)", query, re.IGNORECASE):
        close_application(m.group(1).strip())
        return

    elif "increase volume" in query or "volume up" in query:
        change_system_volume("up")
        return
    elif "decrease volume" in query or "volume down" in query:
        change_system_volume("down")
        return
    elif "mute" in query and "volume" in query:
        change_system_volume("mute")
        return
    elif "unmute" in query and "volume" in query:
        change_system_volume("unmute")
        return
    elif m := re.search(r"(?:google|search google for)\s+(.+)", query, re.IGNORECASE):
        perform_browsing(m.group(1).strip())
        return
    elif m := re.search(
        r"(?:detailed search for|scrape|extract about)\s+(.+)",
        query,
        re.IGNORECASE,
    ):
        handle_detailed_web_search(m.group(1).strip())
        return
    elif m := re.search(
        r"(?:summarize|tell me about|what is|who is)\s+(.+?)(?:\s+on web)?$",
        query,
        re.IGNORECASE,
    ):
        perform_web_search_and_summarize(m.group(1).strip())
        return
    elif "screenshot" in query or "capture screen" in query:
        take_screenshot()
        return

    elif re.search(r"\b(exit|quit|goodbye|bye|terminate)\b", query, re.IGNORECASE):
        speak_random(["Goodbye!", "See you later!", "Shutting down."])
        sys.exit(0) # Exit the script


    is_convo_handled = handle_general_conversation_query(query)
    if is_convo_handled:
        return
    
    speak(f"I'm not sure how to handle '{query}'. Shall I search the web for it?")
    web_search_confirm = command()
    if web_search_confirm and any(word in web_search_confirm for word in ["yes", "sure", "okay", "do it"]):
        perform_web_search_and_summarize(query)

def check_reminders():
    """Checks for reminders and triggers alerts."""
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
                    speak(alert_message)

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
    init_db()
    reminder_thread = threading.Thread(target=check_reminders, daemon=True)
    reminder_thread.start()

    wishMe()
    while True:
        query = command()
        process_command(query)
