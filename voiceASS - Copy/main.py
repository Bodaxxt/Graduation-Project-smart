# main.py
import datetime
import os
import sys
import time
import webbrowser
import pyautogui
import pyttsx3 # فقط لـ CLI
import speech_recognition as sr # فقط لـ CLI
import json
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import numpy as np
import psutil
import subprocess
import vlc
import yt_dlp
import requests

# Conditional import for Streamlit to avoid error if not installed when running CLI only
try:
    import streamlit as st # Will be used by app_streamlit.py to display errors
    STREAMLIT_AVAILABLE_FOR_ERROR_HANDLING = True
except ImportError:
    STREAMLIT_AVAILABLE_FOR_ERROR_HANDLING = False


try:
    from googletrans import Translator as GoogleTransService
except ImportError:
    print("googletrans not found, using google_trans_new as fallback. Consider 'pip install googletrans==4.0.0rc1'")
    from google_trans_new import google_translator
    GoogleTransService = google_translator

from googlesearch import search
from bs4 import BeautifulSoup
from gtts import gTTS # For Streamlit TTS in app_streamlit.py
from playsound import playsound # For CLI speak_in_language
import tempfile
from deep_translator import GoogleTranslator

import sqlite3
import threading
import re

# --- Global Variables ---
OPENWEATHERMAP_API_KEY = "661f67d8cc9df0c9e161f20f23eb60d7" 

player = None
opened_processes = {} 
main_recognizer_cli = None 
google_translate_service_instance = None

script_dir = os.path.dirname(os.path.abspath(sys.argv[0] if hasattr(sys, 'frozen') else __file__))

cli_tts_engine = None 
cli_tts_engine_lock = threading.Lock()

is_music_playing_flag = False
music_player_lock = threading.Lock()

last_assistant_question_context_module_level = None 

_speak_mode = "direct"  
_streamlit_speak_capture_list = []

# --- Load Model and Data ---
model = None
tokenizer = None
label_encoder = None
data = None # For intents.json

def load_all_models_and_data():
    global model, tokenizer, label_encoder, data
    
    if model is not None and tokenizer is not None and label_encoder is not None and data is not None:
        print("NLU model and data already loaded.")
        return

    print("Attempting to load NLU model and data...")
    intents_path_local = os.path.join(script_dir, "intents.json")
    model_path_local = os.path.join(script_dir, "chat_model.keras")
    tokenizer_path_local = os.path.join(script_dir, "tokenizer.pkl")
    label_encoder_path_local = os.path.join(script_dir, "label_encoder.pkl")

    try:
        print(f" intents.json path: {intents_path_local}")
        with open(intents_path_local, encoding="utf-8") as file:
            data_loaded = json.load(file)
        print(" intents.json loaded successfully.")
        
        print(f" chat_model.keras path: {model_path_local}")
        model_loaded = load_model(model_path_local)
        print(" chat_model.keras loaded successfully.")
        
        print(f" tokenizer.pkl path: {tokenizer_path_local}")
        with open(tokenizer_path_local, "rb") as f:
            tokenizer_loaded = pickle.load(f)
        print(" tokenizer.pkl loaded successfully.")
        
        print(f" label_encoder.pkl path: {label_encoder_path_local}")
        with open(label_encoder_path_local, "rb") as encoder_file:
            label_encoder_loaded = pickle.load(encoder_file)
        print(" label_encoder.pkl loaded successfully.")
        
        model = model_loaded
        tokenizer = tokenizer_loaded
        label_encoder = label_encoder_loaded
        data = data_loaded
        
        print("NLU model and all data files loaded successfully.")

    except FileNotFoundError as e:
        error_message = f"ERROR - FileNotFoundError in load_all_models_and_data: {e}\nPlease ensure all model/data files are in the script directory: {script_dir}"
        print(error_message)
        if STREAMLIT_AVAILABLE_FOR_ERROR_HANDLING and 'streamlit' in sys.modules:
            st.error(error_message)
            st.stop() 
        else:
            sys.exit(1)
    except Exception as e:
        error_message = f"ERROR - Exception in load_all_models_and_data: {e}"
        print(error_message)
        import traceback
        traceback.print_exc()
        if STREAMLIT_AVAILABLE_FOR_ERROR_HANDLING and 'streamlit' in sys.modules:
            st.error(error_message)
            st.stop()
        else:
            sys.exit(1)

language_map = {
    "english": "en", "french": "fr", "german": "de", "korean": "ko",
    "spanish": "es", "chinese": "zh-cn", "japanese": "ja", "arabic": "ar",
    "hindi": "hi", "russian": "ru", "portuguese": "pt", "italian": "it"
}

# --- TTS and Speech Recognition ---
def set_speak_mode(mode):
    global _speak_mode, _streamlit_speak_capture_list
    if mode in ["direct", "capture"]:
        _speak_mode = mode
        if mode == "capture": 
            _streamlit_speak_capture_list = [] 
    else:
        print(f"Warning: Invalid speak mode '{mode}'. Using 'direct'.")
        _speak_mode = "direct"

def cli_speak(text, wait=False, force_speak=False): # Renamed for clarity, used by CLI and backend logic
    global cli_tts_engine_lock, is_music_playing_flag, cli_tts_engine, _streamlit_speak_capture_list

    # This function is now primarily for actual TTS output or capturing for Streamlit.
    # The decision to speak or capture is handled by _speak_mode.

    if _speak_mode == "capture":
        if text: 
            cleaned_text = re.sub(r'\n+', '\n', str(text)).strip() 
            if cleaned_text:
                _streamlit_speak_capture_list.append(cleaned_text)
        return

    # Direct speaking mode (CLI)
    if is_music_playing_flag and not force_speak:
        print(f"[CLI Speech Suppressed] Music playing. Suppressed: '{text}'")
        return

    engine_instance = initialize_cli_tts_engine()
    if engine_instance:
        with cli_tts_engine_lock:
            try: engine_instance.say(text); engine_instance.runAndWait()
            except RuntimeError: 
                print(f"CLI TTS Runtime Error. Reinitializing engine.")
                cli_tts_engine = None; engine_instance_retry = initialize_cli_tts_engine()
                if engine_instance_retry:
                    try: engine_instance_retry.say(text); engine_instance_retry.runAndWait()
                    except Exception as e_retry: print(f"CLI TTS Error on retry: {e_retry}")
                else: print("CLI TTS reinitialization failed.")
            except Exception as e: print(f"Error during CLI speech: {e}")
        if wait: time.sleep(0.5)
    else: print(f"CLI TTS SKIPPED (engine not initialized): {text}")


def cli_speak_random(phrases, wait=False, force_speak=False):
    cli_speak(random.choice(phrases), wait=wait, force_speak=force_speak)

def initialize_cli_tts_engine():
    global cli_tts_engine # For pyttsx3
    if cli_tts_engine is None:
        with cli_tts_engine_lock: 
            if cli_tts_engine is None:
                try:
                    # Attempt to initialize COM for pyttsx3 if on Windows
                    if sys.platform == "win32":
                        import pythoncom
                        pythoncom.CoInitialize()
                    engine = pyttsx3.init("sapi5") 
                    if not engine: print("CLI TTS engine pyttsx3.init() failed."); cli_tts_engine = None; return None
                    voices = engine.getProperty('voices')
                    if voices: engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
                    else: print("No TTS voices found for CLI.")
                    rate = engine.getProperty('rate'); engine.setProperty('rate', max(50, rate - 70)) 
                    volume = engine.getProperty('volume'); engine.setProperty('volume', min(1.0, volume + 0.25)) 
                    cli_tts_engine = engine
                # except ImportError as e_com:
                #     print(f"pywin32 'pythoncom' not found, CoInitialize skipped. TTS might fail in threads: {e_com}")
                #     # Continue trying to init pyttsx3 anyway
                #     try: # Try again without CoInitialize if pythoncom import failed
                #         engine = pyttsx3.init("sapi5")
                #         if not engine: print("CLI TTS engine pyttsx3.init() failed (after com error)."); cli_tts_engine = None; return None
                #         # ... (rest of the setup)
                #         cli_tts_engine = engine
                #     except Exception as e_tts_no_com:
                #         print(f"Error initializing CLI TTS engine (after com error): {e_tts_no_com}"); cli_tts_engine = None

                except Exception as e: 
                    print(f"Error initializing CLI TTS engine: {e}")
                    cli_tts_engine = None
    return cli_tts_engine

def initialize_cli_recognizer():
    global main_recognizer_cli
    if main_recognizer_cli is None:
        main_recognizer_cli = sr.Recognizer()
    return main_recognizer_cli

def cli_command(prompt_text=None, force_prompt=False): # For CLI voice input
    current_recognizer = initialize_cli_recognizer()
    if prompt_text: cli_speak(prompt_text, force_speak=True if force_prompt else False)
    try:
        with sr.Microphone() as source:
            try: current_recognizer.adjust_for_ambient_noise(source, duration=0.5)
            except Exception as e_adjust: print(f"Could not adjust for ambient noise (CLI): {e_adjust}")
            print("Listening (CLI)...", end="", flush=True)
            current_recognizer.pause_threshold = 0.8 
            audio = current_recognizer.listen(source, timeout=7, phrase_time_limit=12)
            print("\rRecognizing (CLI)...", end="", flush=True)
            query = current_recognizer.recognize_google(audio, language='en-US')
            print(f"\rUser said (CLI): {query}\n"); return query.lower()
    except sr.WaitTimeoutError: print("\rNo speech detected (CLI)."); return None
    except sr.UnknownValueError: print("\rCould not understand audio (CLI)."); return "unintelligible"
    except sr.RequestError as e: print(f"\rSpeech service request error (CLI); {e}"); cli_speak_random(["My speech recognition is offline.", "Connection issue with speech service."], force_speak=True); return None
    except OSError as e_os: print(f"\rOSError with microphone (CLI): {e_os}. Check microphone."); cli_speak("Microphone access error. Check connection.", force_speak=True); return None
    except Exception as e: print(f"\rUnexpected error in CLI command(): {e}"); import traceback; traceback.print_exc(); return None

# --- Database Initialization (Reminders Only) ---
DATABASE_NAME = os.path.join(script_dir, "assistant_data.db")
def init_db():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME, timeout=10, check_same_thread=False) 
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY AUTOINCREMENT, description TEXT NOT NULL, datetime TEXT NOT NULL, repeat TEXT NOT NULL)''') 
        conn.commit()
    except sqlite3.Error as e: print(f"Database initialization error: {e}"); sys.exit(1) 
    finally:
        if conn: conn.close()

# --- Reminder System Functions ---
def add_reminder_db(desc, dt_string, repeat_type): # Keep as is
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME, timeout=10, check_same_thread=False); c = conn.cursor()
        c.execute("INSERT INTO reminders (description, datetime, repeat) VALUES (?, ?, ?)", (desc, dt_string, repeat_type))
        conn.commit()
    except sqlite3.Error as e: print(f"DB Error adding reminder: {e}")
    finally: 
        if conn: conn.close()

def remove_reminder_db(reminder_id): # Keep as is
    conn = None; deleted_count = 0
    try:
        conn = sqlite3.connect(DATABASE_NAME, timeout=10, check_same_thread=False); c = conn.cursor()
        c.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,)); deleted_count = c.rowcount
        conn.commit()
    except sqlite3.Error as e: print(f"DB Error removing reminder: {e}")
    finally: 
        if conn: conn.close()
    return deleted_count > 0

def get_reminders_db(): # Keep as is
    conn = None; reminders = []
    try:
        conn = sqlite3.connect(DATABASE_NAME, timeout=10, check_same_thread=False); c = conn.cursor()
        c.execute("SELECT id, description, datetime, repeat FROM reminders ORDER BY datetime"); reminders = c.fetchall()
    except sqlite3.Error as e: print(f"DB Error getting reminders: {e}")
    finally:
        if conn: conn.close()
    return reminders

def parse_month(text): # Keep as is
    months = { "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12 }
    text_lower = text.lower().strip()
    for name, num in months.items():
        if name in text_lower: return num
    match_num = re.search(r'\b([1-9]|1[0-2])\b', text_lower) 
    if match_num: return int(match_num.group(1))
    return None

def get_cli_reminder_detail(prompt_message, retries=2): # For CLI flows
    response = cli_command(prompt_text=prompt_message, force_prompt=True) 
    for attempt in range(retries):
        if response and response != "unintelligible": return response.strip() 
        elif response == "unintelligible":
            if attempt < retries - 1: cli_speak_random(["I didn't quite catch that. Could you say it again?", "Sorry, one more time?"], force_speak=True); response = cli_command() 
            else: cli_speak("I'm still having trouble understanding. Let's skip this part for now.", force_speak=True); return None
        elif response is None: 
             if attempt < retries - 1: cli_speak("I didn't hear anything. Please tell me.", force_speak=True); response = cli_command()
             else: cli_speak("No input received. Skipping this detail.", force_speak=True); return None
    return None

def check_reminders(): # Modified to use cli_speak and _speak_mode
    print("Reminder checking thread started.")
    while True:
        try:
            now = datetime.datetime.now(); reminders_to_check = get_reminders_db() 
            for r_id, desc, dt_str, repeat in reminders_to_check:
                try: reminder_time = datetime.datetime.fromisoformat(dt_str)
                except ValueError: print(f"Error parsing reminder datetime: {dt_str} for ID {r_id}. Skipping."); continue
                if now >= reminder_time:
                    print(f"[Reminder Alert] ID {r_id}: {desc} at {dt_str}, Repeat: {repeat}")
                    if _speak_mode == "direct": cli_speak(f"Reminder: {desc}", force_speak=True)
                    conn = None 
                    try:
                        conn = sqlite3.connect(DATABASE_NAME, timeout=10, check_same_thread=False); c = conn.cursor()
                        if repeat.lower() == 'once': c.execute("DELETE FROM reminders WHERE id = ?", (r_id,)); print(f"Reminder ID {r_id} (once) removed.")
                        else: 
                            next_reminder_time = reminder_time; valid_repeat = False
                            if repeat.lower() == 'daily': 
                                while next_reminder_time <= now: next_reminder_time += datetime.timedelta(days=1)
                                valid_repeat = True
                            elif repeat.lower() == 'weekly': 
                                while next_reminder_time <= now: next_reminder_time += datetime.timedelta(weeks=1)
                                valid_repeat = True
                            if valid_repeat: c.execute("UPDATE reminders SET datetime = ? WHERE id = ?", (next_reminder_time.isoformat(), r_id)); print(f"Reminder ID {r_id} ({repeat}) rescheduled to {next_reminder_time.isoformat()}.")
                            else: c.execute("DELETE FROM reminders WHERE id = ?", (r_id,)); print(f"Reminder ID {r_id} (unknown repeat '{repeat}') removed.")
                        conn.commit()
                    except sqlite3.Error as e_sql_update: print(f"DB Error updating/deleting reminder ID {r_id}: {e_sql_update}")
                    finally:
                        if conn: conn.close()
            time.sleep(30) 
        except sqlite3.Error as e_sql_main: print(f"SQLite error in check_reminders: {e_sql_main}"); time.sleep(60) 
        except Exception as e: print(f"Unexpected error in check_reminders: {e}"); import traceback; traceback.print_exc(); time.sleep(60) 

def manage_reminders_flow(): # Uses cli_speak and get_cli_reminder_detail
    global last_assistant_question_context_module_level
    last_assistant_question_context_module_level = None 
    cli_speak("What would you like to do with reminders? You can add, view, or remove a reminder.", force_speak=True)
    
    action = cli_command() 
    if 'streamlit' in sys.modules and "get_chatbot_response" in [frame.f_code.co_name for frame in sys._current_frames().values() for frame in [frame] + list(iter_frames(frame.f_back))]:
        if not action: # If called from Streamlit context and no voice input (cli_command returned None)
            cli_speak("For reminder management in the web interface, please use specific commands like 'add reminder to call mom tomorrow at 3 PM', 'show my reminders', or 'delete reminder number 5'. The interactive flow works best in the command line version.")
            return
            
    if not action or action == "unintelligible": cli_speak("I didn't catch that. Please try again.", force_speak=True); return
    action = action.lower()

    if "add" in action:
        desc = get_cli_reminder_detail("What is the reminder about?")
        if not desc: cli_speak("Description cannot be empty. Cancelling.", force_speak=True); return
        parsed_date_obj = None
        for _ in range(3): 
            date_input = get_cli_reminder_detail("What date? (e.g., 'tomorrow', 'next Friday', 'July 25th')")
            if not date_input: cli_speak("Date not provided. Cancelling.", force_speak=True); return
            today = datetime.date.today(); date_input_lower = date_input.lower()
            if "today" in date_input_lower: parsed_date_obj = today; break
            elif "tomorrow" in date_input_lower: parsed_date_obj = today + datetime.timedelta(days=1); break
            elif "next" in date_input_lower:
                days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                try:
                    target_day_name = next(day for day in days_of_week if day in date_input_lower)
                    days_ahead = days_of_week.index(target_day_name) - today.weekday()
                    if days_ahead <= 0: days_ahead += 7
                    parsed_date_obj = today + datetime.timedelta(days=days_ahead); break
                except StopIteration: pass
            try: parsed_date_obj = datetime.datetime.strptime(date_input, "%Y-%m-%d").date(); break
            except ValueError: pass
            try:
                match_mdy = re.search(r"(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?", date_input, re.IGNORECASE)
                if match_mdy:
                    month_name, day_num_str, year_str = match_mdy.groups(); month_num = parse_month(month_name)
                    if month_num:
                        day_num = int(day_num_str); year_num = int(year_str) if year_str else today.year
                        parsed_date_obj = datetime.date(year_num, month_num, day_num)
                        if not year_str and parsed_date_obj < today : parsed_date_obj = datetime.date(today.year + 1, month_num, day_num)
                        break
            except: pass
            cli_speak("Could not understand date. Try 'Month Day Year' or 'YYYY-MM-DD'.", force_speak=True)
        if not parsed_date_obj: cli_speak("Failed to parse date. Cancelling.", force_speak=True); return
        date_str = parsed_date_obj.isoformat()

        time_obj = None
        for _ in range(3):
            time_input_raw = get_cli_reminder_detail("What time? (e.g., 9 AM, 2:30 PM, 14:00)")
            if not time_input_raw: cli_speak("Time not provided. Cancelling.", force_speak=True); return
            time_input = time_input_raw.strip().upper().replace("A.M.", "AM").replace("P.M.", "PM")
            if time_input == "12 AM" or time_input == "12AM": time_obj = datetime.time(0,0); break
            if time_input == "12 PM" or time_input == "12PM": time_obj = datetime.time(12,0); break
            if "NOON" in time_input: time_obj = datetime.time(12,0); break
            if "MIDNIGHT" in time_input: time_obj = datetime.time(0,0); break
            match_24h_with_ampm = re.match(r"(\d{1,2}):(\d{2})\s*(AM|PM)", time_input)
            if match_24h_with_ampm:
                hour_str, minute_str, ampm_str = match_24h_with_ampm.groups(); hour_val = int(hour_str)
                if hour_val > 12 or (hour_val == 12 and ampm_str == "AM"): 
                     if hour_val == 12 and ampm_str == "AM": hour_val = 0 
                     time_input = f"{hour_val:02d}:{minute_str}" 
            time_formats = ["%I:%M %p", "%I%p", "%H:%M"] 
            time_input_processed = time_input.replace(".", ":").replace(" ", "")
            for fmt in time_formats:
                try: time_obj = datetime.datetime.strptime(time_input_processed, fmt).time(); break
                except ValueError: continue
            if time_obj: break
            cli_speak("Invalid time. Try HH:MM, H AM/PM (e.g. 9 AM, 2:30 PM, 14:00).", force_speak=True)
        if not time_obj: cli_speak("Failed to parse time. Cancelling.", force_speak=True); return
        time_str_iso = time_obj.strftime("%H:%M:%S")
        repeat_input_raw = get_cli_reminder_detail("Repeat? (once, daily, weekly, none)"); repeat_type = "once"
        if repeat_input_raw:
            repeat_input_lower = repeat_input_raw.lower()
            if "daily" in repeat_input_lower: repeat_type = "daily"
            elif "weekly" in repeat_input_lower: repeat_type = "weekly"
        try:
            final_dt_obj = datetime.datetime.fromisoformat(f"{date_str}T{time_str_iso}")
            if final_dt_obj <= datetime.datetime.now() + datetime.timedelta(minutes=1): cli_speak("Time is in past/too soon.", force_speak=True); return
            add_reminder_db(desc, final_dt_obj.isoformat(), repeat_type)
            cli_speak(f"Set {repeat_type} reminder: {desc} on {final_dt_obj.strftime('%A, %B %d at %I:%M %p')}.", force_speak=True)
        except Exception as e: print(f"Error setting reminder: {e}"); cli_speak("Sorry, error setting reminder.", force_speak=True)
    elif "view" in action or "show" in action or "list" in action:
        reminders = get_reminders_db()
        if not reminders: cli_speak("You have no reminders set.", force_speak=True); return
        cli_speak("Here are your reminders:", force_speak=True)
        for r_id, desc_db, dt_str, repeat in reminders: 
            try: dt_obj = datetime.datetime.fromisoformat(dt_str); cli_speak(f"Number {r_id}: {desc_db}, on {dt_obj.strftime('%A, %B %d at %I:%M %p')}. Repeats: {repeat}.", force_speak=True)
            except ValueError: cli_speak(f"Number {r_id}: {desc_db}, at invalid time '{dt_str}'. Repeats: {repeat}.", force_speak=True)
    elif "remove" in action or "delete" in action:
        reminders = get_reminders_db()
        if not reminders: cli_speak("You have no reminders to remove.", force_speak=True); return
        cli_speak("Here are your reminders. Which one to remove? Say the number or part of the description.", force_speak=True)
        for r_id, desc_db, dt_str, repeat in reminders: 
             try: dt_obj = datetime.datetime.fromisoformat(dt_str); cli_speak(f"Number {r_id}: {desc_db} ({dt_obj.strftime('%b %d, %I:%M%p')})", force_speak=True)
             except ValueError: cli_speak(f"Number {r_id}: {desc_db} (invalid date)", force_speak=True)
        id_to_remove_str_raw = cli_command()
        if not id_to_remove_str_raw or id_to_remove_str_raw == "unintelligible": cli_speak("No selection made. Cancelling removal.", force_speak=True); return
        id_to_remove_str = id_to_remove_str_raw.strip().lower(); removed_flag = False
        if id_to_remove_str.isdigit():
            target_id = int(id_to_remove_str)
            if any(r[0] == target_id for r in reminders): 
                if remove_reminder_db(target_id): cli_speak(f"Reminder number {target_id} removed.", force_speak=True); removed_flag = True
            else: cli_speak(f"No reminder found with number {target_id}.", force_speak=True)
        if not removed_flag:
            matched_reminders = [r for r_id_match, desc_db_match, _, _ in reminders if id_to_remove_str in desc_db_match.lower() for r in [(r_id_match, desc_db_match)]] 
            if not matched_reminders: cli_speak(f"Could not find a reminder matching '{id_to_remove_str_raw}'.", force_speak=True)
            elif len(matched_reminders) == 1:
                r_id_match, desc_match = matched_reminders[0]
                cli_speak(f"Found: Number {r_id_match}, '{desc_match}'. Remove it? (yes/no)", force_speak=True)
                confirm = cli_command()
                if confirm and confirm.lower() in ["yes", "yeah", "ok", "yep"]:
                    if remove_reminder_db(r_id_match): cli_speak(f"Reminder '{desc_match}' removed.", force_speak=True); removed_flag = True
            else: 
                cli_speak("Multiple matches. Use the number:", force_speak=True)
                for r_id_multi, desc_multi in matched_reminders: cli_speak(f"Number {r_id_multi}: {desc_multi}", force_speak=True)
        if not removed_flag and not id_to_remove_str.isdigit(): cli_speak(f"Failed to remove reminder '{id_to_remove_str_raw}'. Try number.", force_speak=True)
    else: cli_speak("Sorry, I didn't understand that reminder action.", force_speak=True)


# --- Assistant Core Functions (Using cli_speak for output) ---
def get_weather_info(city_name): 
    global last_assistant_question_context_module_level; last_assistant_question_context_module_level = None 
    if OPENWEATHERMAP_API_KEY == "YOUR_ACTUAL_OPENWEATHERMAP_API_KEY" or not OPENWEATHERMAP_API_KEY or len(OPENWEATHERMAP_API_KEY) < 30:
        cli_speak("Weather service not configured. Add API key.", force_speak=True); return
    params = {'q': city_name.strip(), 'appid': OPENWEATHERMAP_API_KEY, 'units': 'metric'}
    try:
        response = requests.get("http://api.openweathermap.org/data/2.5/weather?", params=params, timeout=10) 
        response.raise_for_status(); weather_data = response.json()
        if str(weather_data.get("cod")) == "404": cli_speak(f"Sorry, {weather_data.get('message', 'city not found')}.", force_speak=True); return
        if weather_data.get("main") and weather_data.get("weather"):
            main, weather = weather_data["main"], weather_data["weather"][0]
            wind_spd = f"{weather_data.get('wind', {}).get('speed', 0) * 3.6:.1f} km/h"
            report = f"In {weather_data.get('name', city_name.capitalize())}: Temp {main.get('temp')}°C (feels like {main.get('feels_like')}°C), {weather.get('description')}. Humidity {main.get('humidity')}%. Wind {wind_spd}."
            cli_speak(report, force_speak=True)
        else: cli_speak(f"Unexpected weather response for {city_name}.", force_speak=True)
    except requests.exceptions.HTTPError as http_err:
        print(f"Weather HTTP error: {http_err}")
        if http_err.response.status_code == 401: cli_speak("Weather API key invalid or not authorized.", force_speak=True)
        else: cli_speak(f"Weather service error ({http_err.response.status_code}).", force_speak=True)
    except requests.exceptions.RequestException as e: print(f"Weather request error: {e}"); cli_speak("Can't connect to weather service.", force_speak=True)
    except Exception as e: print(f"Weather processing error: {e}"); cli_speak("Error fetching weather.", force_speak=True)

def vlc_state_monitor(): 
    global player, is_music_playing_flag, music_player_lock; last_known_state = None
    while True:
        try:
            with music_player_lock:
                if player:
                    current_state = player.get_state()
                    if current_state != last_known_state: last_known_state = current_state
                    if current_state in [vlc.State.Ended, vlc.State.NothingSpecial, vlc.State.Error, vlc.State.Stopped, vlc.State.Paused]:
                        if is_music_playing_flag: is_music_playing_flag = False
                    elif current_state == vlc.State.Playing:
                        if not is_music_playing_flag: is_music_playing_flag = True
                elif is_music_playing_flag: is_music_playing_flag = False; last_known_state = None 
            time.sleep(0.5) 
        except Exception as e_monitor: print(f"Error in vlc_state_monitor: {e_monitor}"); time.sleep(5) 

def search_youtube(query): 
    ydl_opts = {'format': 'bestaudio/best', 'noplaylist': True, 'quiet': True, 'default_search': 'ytsearch1:', 'extract_flat': 'discard_in_playlist', 'forcejson': True, 'skip_download': True }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(query, download=False)
            if result and 'entries' in result and result['entries']: return result['entries'][0].get('webpage_url'), result['entries'][0].get('title', 'Unknown Title') 
            elif result and 'webpage_url' in result: return result.get('webpage_url'), result.get('title', 'Unknown Title')
        print(f"No YouTube video for: {query}"); return None, None
    except yt_dlp.utils.DownloadError as e: print(f"yt-dlp DL error for '{query}': {e}"); return None, None
    except Exception as e: print(f"YouTube search error for '{query}': {e}"); return None, None

def get_best_audio_url(video_url):
    if not video_url: return None
    ydl_opts = {'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best', 'quiet': True, 'noplaylist': True, 'skip_download': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: 
            info = ydl.extract_info(video_url, download=False)
            if 'url' in info and info['url'].startswith('http') and info.get('acodec') != 'none' and not info.get('formats'):
                print(f"Extracted direct URL (top-level): {info['url']} for {video_url}"); return info['url']
            if 'formats' in info:
                audio_formats = [f for f in info['formats'] if f.get('acodec') != 'none' and f.get('vcodec') == 'none' and f.get('url')]
                if not audio_formats: audio_formats = [f for f in info['formats'] if f.get('acodec') != 'none' and f.get('url')]
                if audio_formats:
                    for ext_pref in ['m4a', 'webm', 'ogg', 'mp3']:
                        for f_format in audio_formats:
                            if f_format.get('ext') == ext_pref: print(f"Extracted URL (format {f_format.get('format_id', 'N/A')}, ext {ext_pref}): {f_format['url']} for {video_url}"); return f_format['url']
                    print(f"Extracted URL (first audio format): {audio_formats[0]['url']} for {video_url}"); return audio_formats[0]['url']
            print(f"Could not reliably extract audio URL for: {video_url}"); return None
    except Exception as e: print(f"Error getting audio URL for '{video_url}': {e}"); return None

def play_song(song_query):
    global player, is_music_playing_flag, music_player_lock 
    with music_player_lock:
        if player:
            if player.is_playing() or player.get_state() == vlc.State.Paused : player.stop()
            player.release(); player = None 
        is_music_playing_flag = False 
    cli_speak_random([f"Looking for {song_query}...", f"Finding {song_query}..."], force_speak=True)
    video_url, video_title = search_youtube(song_query)
    if not video_url: cli_speak(f"Sorry, couldn't find '{song_query}'.", force_speak=True); return
    cli_speak(f"Found '{video_title}'. Getting audio...", force_speak=True)
    playurl = get_best_audio_url(video_url)
    if not playurl: cli_speak(f"Found '{video_title}', but trouble getting audio stream.", force_speak=True); return
    print(f"Attempting to play URL: {playurl}")
    with music_player_lock: 
        try:
            vlc_instance = None
            try:
                instance_args = ['--no-xlib', '--quiet', '--verbose=-1', '--no-qt-privacy-ask', '--no-metadata-network-access']
                vlc_instance = vlc.Instance(instance_args)
                if not vlc_instance: raise vlc.VLCException("VLC instance is None")
                player = vlc_instance.media_player_new() 
                if not player: raise vlc.VLCException("VLC media_player_new is None")
            except Exception as e_vlc_init: print(f"VLC Init error: {e_vlc_init}"); cli_speak("Audio player error. Check VLC install.", force_speak=True); return
            media = vlc_instance.media_new(playurl); player.set_media(media); player.play()
            playback_started = False
            for _ in range(10): 
                time.sleep(0.5); current_vlc_state = player.get_state()
                if current_vlc_state == vlc.State.Playing: playback_started = True; break
                if current_vlc_state in [vlc.State.Error, vlc.State.Ended, vlc.State.Stopped]: print(f"[Music] Playback failed/stopped early. State: {current_vlc_state}"); break
            if playback_started:
                is_music_playing_flag = True; print(f"[Music] Playing: {video_title}"); cli_speak(f"Now playing: {video_title}.", force_speak=True)
            else:
                current_vlc_state = player.get_state()
                print(f"[Music] Failed to play '{video_title}'. Final VLC State: {current_vlc_state}")
                if player: player.release(); player = None 
                is_music_playing_flag = False; cli_speak(f"Tried to play {video_title}, but failed. Player state: {current_vlc_state}", force_speak=True)
        except Exception as e_playback: 
            print(f"Playback error for '{song_query}': {e_playback}")
            if player: player.release(); player = None 
            is_music_playing_flag = False; cli_speak_random(["Oops, song playback error.", "Error with music."], force_speak=True)

def next_song(): 
    global player
    if player and (player.is_playing() or player.get_state() == vlc.State.Paused):
        cli_speak("Skipping to next... (Note: True playlist 'next' for YouTube streams is complex.)", force_speak=True)
        if player.is_playing(): player.stop() 
    else:
        cli_speak("No music is playing to skip.", force_speak=True)

def stop_song():
    global player, is_music_playing_flag, music_player_lock; response_text = ""; action_taken = False
    with music_player_lock:
        if player:
            if player.is_playing() or player.get_state() == vlc.State.Paused: player.stop(); action_taken = True
            player.release(); player = None; print("[Music] Player stopped/released.")
        response_text = random.choice(["Music stopped.", "Playback halted."]) if is_music_playing_flag or action_taken else random.choice(["No music playing.", "Nothing to stop."])
        is_music_playing_flag = False 
    cli_speak(response_text, force_speak=True)

def perform_web_search_and_summarize(query_text): 
    global last_assistant_question_context_module_level
    cli_speak_random([f"Looking up '{query_text}'...", f"Searching for '{query_text}'..."], force_speak=True)
    search_results_list = []
    try:
        search_results_iter = search(query_text, num_results=1, lang='en', sleep_interval=1) 
        search_results_list = list(search_results_iter)
    except Exception as e: cli_speak(f"Web search issue: {e}", force_speak=True); return
    if not search_results_list: cli_speak(f"No web results for '{query_text}'.", force_speak=True); return
    first_url = search_results_list[0]; print(f"Summarizing: {first_url}")
    cli_speak("Found a page, trying to summarize...", force_speak=True)
    page_content_summary = None
    try:
        response = requests.get(first_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15); response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for el_type in ['script','style','nav','footer','aside','header','form','button','img','iframe','link','meta','noscript','a','input','select','textarea','figure','figcaption','svg','path','video','audio','canvas','map','area']:
            for el in soup.find_all(el_type): el.decompose()
        paragraphs = [p.get_text(" ",strip=True) for p in soup.find_all('p') if p.get_text(strip=True) and len(p.get_text(strip=True).split()) > 10]
        if paragraphs:
            full_text = ' '.join(paragraphs); sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', full_text)
            summary_parts, wc, max_w = [], 0, 80
            for s in sentences: 
                if not s.strip(): continue
                summary_parts.append(s.strip()); wc += len(s.split())
                if wc > max_w: break
            page_content_summary = ' '.join(summary_parts) if summary_parts else "Could not extract meaningful summary."
        else: page_content_summary = "No suitable paragraphs found for summary."
    except Exception as e: print(f"Summarization error for {first_url}: {e}"); page_content_summary = "Issue summarizing page."
    cli_speak(f"Summary: {page_content_summary}" if page_content_summary else "Could not summarize.", force_speak=True)
    cli_speak("Open this page for more details?", force_speak=True)
    last_assistant_question_context_module_level = {'type': 'open_page_confirm', 'url': first_url}

def web_search_detailed(query, num_results=1): 
    try: return list(search(query, num_results=num_results, lang='en', sleep_interval=1))
    except Exception as e: print(f"Detailed search error: {e}"); return []

def get_page_content_detailed(url): 
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10); response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for el in soup(['script', 'style', 'nav', 'footer', 'iframe', 'aside', 'header', 'form', 'button', 'img', 'a']): el.decompose()
        texts = [el.get_text(" ", strip=True) for tag in ['article','main','section','div','p'] for el in soup.find_all(tag) if el.get_text(strip=True) and len(el.get_text(strip=True).split()) > 10]
        if texts:
            content = ". ".join(texts); sentences = [s.strip() for s in content.split('.') if s.strip()]
            keywords = ['study', 'research', 'found', 'show', 'reveal', 'important', 'key', 'conclusion']
            imp_sentences = sentences[:2] 
            for s in sentences[2:8]: 
                if any(k in s.lower() for k in keywords) and len(imp_sentences) < 5: imp_sentences.append(s)
            return ('. '.join(imp_sentences[:5]) + '.').strip() if imp_sentences else "No specific important sentences found."
        return "No suitable content for detailed summary."
    except Exception as e: print(f"Error getting detailed content for {url}: {e}"); return f"Could not fetch/process {url}."

def handle_detailed_web_search(query):
    global last_assistant_question_context_module_level
    cli_speak(f"Okay, performing a detailed search for {query}...", force_speak=True)
    results = web_search_detailed(query, num_results=1)
    if not results: cli_speak("No results for detailed search.", force_speak=True); return
    top_url = results[0]; print(f"\nDetailed search top URL: {top_url}")
    cli_speak(f"Found a page. Extracting detailed content...", force_speak=True)
    content_summary = get_page_content_detailed(top_url)
    if content_summary and not content_summary.startswith("Could not") and not content_summary.startswith("An error"):
        print(f"\nDetailed Summary: {content_summary}\n")
        cli_speak(f"Here's the detailed information: {content_summary[:300]}...", force_speak=True) 
        cli_speak("Open the page for all details?", force_speak=True)
        last_assistant_question_context_module_level = {'type': 'open_page_confirm', 'url': top_url}
    else: print(f"Detailed content extraction failed: {content_summary}"); cli_speak(f"Sorry, I {content_summary.lower() if content_summary else 'could not extract detailed content.'}", force_speak=True)

def predict_intent_from_text(text): 
    global model, tokenizer, label_encoder # Ensure these are accessible
    if not all([model, tokenizer, label_encoder]):
        print("NLU models not loaded. Cannot predict intent.")
        return None, 0.0
    try:
        seq = tokenizer.texts_to_sequences([text]); maxlen_model = 20 
        if hasattr(model, 'input_shape') and isinstance(model.input_shape, (tuple, list)) and len(model.input_shape) > 1 and model.input_shape[1] is not None: maxlen_model = model.input_shape[1]
        padded = pad_sequences(seq, maxlen=maxlen_model, padding='post', truncating='post')
        prediction = model.predict(padded, verbose=0)
        tag_index = np.argmax(prediction[0]); tag = label_encoder.inverse_transform([tag_index])[0]
        return tag, float(prediction[0][tag_index])
    except Exception as e: print(f"Predict intent error: {e}"); return None, 0.0

def handle_general_conversation_query(query_text): 
    global data # Ensure data (from intents.json) is accessible
    if not data:
        print("Intents data not loaded. Cannot handle general conversation.")
        return False
        
    tag, confidence = predict_intent_from_text(query_text)
    if tag: 
        print(f"Intent: {tag}, Confidence: {confidence:.3f}")
        if confidence > 0.65: 
            for intent_data in data['intents']:
                if intent_data['tag'] == tag:
                    cli_speak(random.choice(intent_data['responses']), force_speak=tag in ["greetings", "thanks", "goodbye", "help", "facts"])
                    return True 
    return False 

# --- System Interaction Functions ---
def wishMe(): cli_speak(f"Good {('Morning' if 0 <= datetime.datetime.now().hour < 12 else 'Afternoon' if 12 <= datetime.datetime.now().hour < 18 else 'Evening')} Abdallah! How can I help?", force_speak=True)
def get_location(): 
    try:
        response_data = requests.get('https://ipinfo.io/json', timeout=7).json() # Renamed data to response_data
        cli_speak(f"Looks like you're near {response_data.get('city', 'unknown city')}, {response_data.get('region', 'unknown region')}.", force_speak=True)
    except Exception as e: print(f"Location error: {e}"); cli_speak("Couldn't determine location.", force_speak=True)

def speak_in_language(text_to_speak, lang_code): # For CLI playsound
    if is_music_playing_flag: print(f"[Speak In Lang Suppressed] Music playing."); return 
    temp_filename = None
    try:
        tts = gTTS(text=text_to_speak, lang=lang_code, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', dir=tempfile.gettempdir()) as fp: temp_filename = fp.name
        tts.save(temp_filename); playsound(temp_filename); print(f"Spoke '{text_to_speak}' in {lang_code}.")
    except Exception as e: print(f"Speak in lang error for '{lang_code}': {e}"); cli_speak("Error speaking in that language.", force_speak=True)
    finally:
        if temp_filename and os.path.exists(temp_filename):
            try: os.remove(temp_filename)
            except Exception as e_remove: print(f"Error removing temp audio: {e_remove}")

def open_social_media(platform_name): 
    urls = {"facebook":"https://facebook.com","twitter":"https://twitter.com","youtube":"https://youtube.com","instagram":"https://instagram.com","linkedin":"https://linkedin.com"}
    url = urls.get(platform_name.lower().strip())
    if url: cli_speak(f"Opening {platform_name.capitalize()}.", force_speak=True); webbrowser.open(url)
    else: cli_speak(f"Don't have a link for {platform_name}.", force_speak=True)

def open_application(app_name_query): 
    global opened_processes; app_map = {"notepad":"notepad.exe", "calculator":"calc.exe", "chrome":"chrome.exe"} 
    cmd, name, exe = None, None, None; query_lower = app_name_query.lower().strip()
    for n, c in app_map.items():
        if n in query_lower: cmd, name, exe = c, n.capitalize(), c; break
    if cmd:
        cli_speak(f"Opening {name}.", force_speak=True)
        try:
            proc = subprocess.Popen(cmd); opened_processes[exe.lower()] = proc
        except Exception as e: cli_speak(f"Error opening {name}: {e}", force_speak=True)
    else: cli_speak(f"Don't know how to open '{app_name_query}'.", force_speak=True)

def close_application(app_name_query): 
    global opened_processes; query_lower = app_name_query.lower().strip()
    app_exe_map = {"notepad":"notepad.exe", "calculator":"calc.exe", "chrome":"chrome.exe"}
    exe_to_close = next((exe for name, exe in app_exe_map.items() if name in query_lower), query_lower if query_lower.endswith(".exe") else query_lower + ".exe" if sys.platform == "win32" else query_lower)
    if sys.platform == "win32" and exe_to_close:
        try:
            res = subprocess.run(["taskkill", "/F", "/IM", exe_to_close], check=True, capture_output=True, text=True)
            if "SUCCESS" in res.stdout: cli_speak(f"Closed {app_name_query}.", force_speak=True)
            else: cli_speak(f"Could not find or close {app_name_query}. {res.stderr or res.stdout}", force_speak=True)
        except Exception as e: cli_speak(f"Error closing {app_name_query}: {e}", force_speak=True)
    elif exe_to_close.lower() in opened_processes:
        proc = opened_processes.pop(exe_to_close.lower()); proc.terminate()
        try: proc.wait(timeout=2); cli_speak(f"Closed {app_name_query}.", force_speak=True)
        except: proc.kill(); cli_speak(f"Force closed {app_name_query}.", force_speak=True)
    else: cli_speak(f"Don't know how to close '{app_name_query}' or it wasn't opened by me.", force_speak=True)

def perform_browsing(search_term): 
    try: webbrowser.open(f"https://www.google.com/search?q={requests.utils.quote(search_term.strip())}"); cli_speak(f"Searching Google for '{search_term}'.", force_speak=True)
    except Exception as e: cli_speak(f"Error opening search: {e}", force_speak=True)

def check_system_condition():
    try:
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk_path = '/'
        if sys.platform == "win32":
            system_root = os.environ.get('SystemRoot', 'C:\\')
            disk_path = os.path.splitdrive(system_root)[0] + os.sep
        try:
            disk = psutil.disk_usage(disk_path)
            disk_report = f"Disk space at {disk_path} is {disk.percent}% used ({disk.free/1024**3:.1f}GB free)." 
        except Exception as e_disk:
            print(f"Error getting disk usage for '{disk_path}': {e_disk}")
            disk_report = "Could not retrieve disk usage information."
        report_parts = [f"CPU at {cpu}%", f"Memory at {mem.percent}% used ({mem.available/1024**3:.1f}GB free)", disk_report]
        if hasattr(psutil, "sensors_battery"): 
            battery = psutil.sensors_battery()
            if battery:
                plugged = "charging" if battery.power_plugged and battery.percent < 100 else ("full" if battery.power_plugged and battery.percent >= 99.5 else "on battery") 
                report_parts.append(f"Battery is at {battery.percent}% ({plugged})")
                if not battery.power_plugged and battery.secsleft not in (psutil.POWER_TIME_UNLIMITED, psutil.POWER_TIME_UNKNOWN, None) and battery.secsleft > 0 :
                    hours, remainder = divmod(battery.secsleft, 3600); minutes, _ = divmod(remainder, 60)
                    report_parts.append(f"About {int(hours)} hours and {int(minutes)} minutes remaining")
        cli_speak("System Status: " + ". ".join(report_parts) + ".", force_speak=True)
    except Exception as e_psutil:
        print(f"Error getting system condition: {e_psutil}")
        cli_speak("I'm having trouble getting the full system status.", force_speak=True)

def translate_text_flow(text_to_translate, target_language_name): 
    global last_assistant_question_context_module_level, google_translate_service_instance 
    if not text_to_translate: cli_speak("What to translate?", force_speak=True); return
    target_lang_lower = target_language_name.lower().strip()
    if target_lang_lower not in language_map: cli_speak(f"Can't translate to {target_language_name}. Supported: {', '.join(list(language_map.keys())[:5])}...", force_speak=True); return
    target_lang_code = language_map[target_lang_lower]; translated_text, used = None, "N/A"
    try: translated_text = GoogleTranslator(source='auto', target=target_lang_code).translate(text_to_translate); used = "DeepTranslator"
    except Exception as e_deep:
        print(f"DeepTranslator error: {e_deep}. Trying fallback.")
        if google_translate_service_instance:
            try: trans_obj = google_translate_service_instance.translate(text_to_translate, dest=target_lang_code); translated_text = trans_obj.text; used = "googletrans"
            except Exception as e_trans: print(f"googletrans error: {e_trans}")
    if not translated_text: cli_speak(f"Trouble translating to {target_language_name}.", force_speak=True); return
    cli_speak(f"In {target_language_name.capitalize()}: '{translated_text}'. (via {used}).", force_speak=True)
    if target_lang_code != 'en': 
        cli_speak(f"Say that in {target_language_name.capitalize()}?", force_speak=True)
        last_assistant_question_context_module_level = {'type': 'say_translation_confirm', 'text': translated_text, 'lang_code': target_lang_code, 'lang_name': target_language_name.capitalize()}
    else: last_assistant_question_context_module_level = None

def change_system_volume(direction, amount=10):
    try:
        if sys.platform == "win32":
            if direction == "up": pyautogui.press("volumeup", presses=amount // 5 if amount > 5 else 1) 
            elif direction == "down": pyautogui.press("volumedown", presses=amount // 5 if amount > 5 else 1)
            elif direction == "mute" or direction == "unmute": pyautogui.press("volumemute") 
            cli_speak(f"Volume {direction}.", force_speak=True)
        elif sys.platform == "darwin": 
            script = ""
            if direction == "up": script = f'set volume output volume (output volume of (get volume settings) + {amount})'
            elif direction == "down": script = f'set volume output volume (output volume of (get volume settings) - {amount})'
            elif direction == "mute": script = 'set volume output muted true'
            elif direction == "unmute": script = 'set volume output muted false'
            if script: subprocess.run(["osascript", "-e", script], check=True)
            cli_speak(f"Volume {direction}.", force_speak=True)
        elif sys.platform.startswith("linux"):
            current_vol_cmd = ["amixer", "sget", "Master"]
            process = subprocess.run(current_vol_cmd, capture_output=True, text=True, check=True)
            match = re.search(r"\[(\d+)%\]", process.stdout); current_percentage = int(match.group(1)) if match else 50
            if direction == "up": new_vol = min(100, current_percentage + amount)
            elif direction == "down": new_vol = max(0, current_percentage - amount)
            elif direction == "mute": subprocess.run(["amixer", "-q", "sset", "Master", "mute"], check=True); cli_speak("Volume muted.", force_speak=True); return
            elif direction == "unmute": subprocess.run(["amixer", "-q", "sset", "Master", "unmute"], check=True); cli_speak("Volume unmuted.", force_speak=True); return
            else: return
            subprocess.run(["amixer", "-q", "sset", "Master", f"{new_vol}%"], check=True)
            cli_speak(f"Volume set to {new_vol} percent.", force_speak=True)
        else: cli_speak("Volume control not supported on this OS yet.", force_speak=True)
    except Exception as e: print(f"Error changing volume: {e}"); cli_speak("Could not change system volume.", force_speak=True)


# --- Main Processing Function ---
def process_command_logic(query, current_context=None):
    global last_assistant_question_context_module_level, is_music_playing_flag, player, opened_processes
    
    last_assistant_question_context_module_level = current_context 
    
    context_handled_this_turn = False
    if last_assistant_question_context_module_level:
        context_type = last_assistant_question_context_module_level.get('type'); lc_query = query.lower().strip() 
        positive_responses = ["yes", "sure", "okay", "do it", "yeah", "yep", "please", "confirm", "affirmative", "go ahead"]
        negative_responses = ["no", "nope", "don't", "cancel", "negative", "stop", "don't do it"]
        if context_type == 'web_search_confirm':
            if any(word in lc_query for word in positive_responses):
                cli_speak(f"Okay, searching for '{last_assistant_question_context_module_level['query']}'.", force_speak=True)
                perform_web_search_and_summarize(last_assistant_question_context_module_level['query']) 
                context_handled_this_turn = True 
            elif any(word in lc_query for word in negative_responses):
                cli_speak("Alright, I won't search.", force_speak=True)
                last_assistant_question_context_module_level = None; context_handled_this_turn = True
        elif context_type == 'open_page_confirm':
            if any(word in lc_query for word in positive_responses):
                cli_speak_random(["Opening the page.", "Alright."], force_speak=True)
                try: webbrowser.open(last_assistant_question_context_module_level['url'])
                except Exception as e: print(f"Error opening browser: {e}") 
                last_assistant_question_context_module_level = None; context_handled_this_turn = True
            elif any(word in lc_query for word in negative_responses):
                cli_speak("Okay, I won't open it.", force_speak=True)
                last_assistant_question_context_module_level = None; context_handled_this_turn = True
        elif context_type == 'say_translation_confirm':
            if any(word in lc_query for word in positive_responses + ["say it"]): 
                speak_in_language(last_assistant_question_context_module_level['text'], last_assistant_question_context_module_level['lang_code'])
            last_assistant_question_context_module_level = None; context_handled_this_turn = True
        
        if context_handled_this_turn: 
            return last_assistant_question_context_module_level 
        
        if not any(word in lc_query for word in positive_responses + negative_responses + ["say it"]):
             print(f"Context '{context_type}' cleared by non-contextual query: '{query}'")
             last_assistant_question_context_module_level = None 

    # --- Direct Command Handling ---    
    play_match = re.search(r"(?:play|listen to|put on|stream)\s+(?:the\s+)?(?:song\s+|music\s+|track\s+)?(.+)", query, re.IGNORECASE)
    if play_match:
        last_assistant_question_context_module_level = None; song_name = play_match.group(1).strip()
        if song_name: play_song(song_name)
        else: 
            cli_speak("Sure, what song or artist?", force_speak=True)
            clarified = cli_command() if _speak_mode == "direct" else None 
            if clarified and clarified != "unintelligible": play_song(clarified.strip())
            elif _speak_mode == "capture" and not clarified: _streamlit_speak_capture_list.append("(Awaiting song name for Streamlit)")
    elif re.search(r"stop\s*(?:the\s*)?(song|music|playback|playing)|stop current track", query, re.IGNORECASE):
        last_assistant_question_context_module_level = None; stop_song()
    elif re.search(r"pause\s*(?:the\s*)?(song|music|playback)|pause playing", query, re.IGNORECASE):
        last_assistant_question_context_module_level = None
        with music_player_lock:
            if player and player.can_pause() and player.is_playing(): player.pause(); is_music_playing_flag = False; cli_speak("Playback paused.", force_speak=True)
            else: cli_speak("Nothing playing to pause.", force_speak=True)
    elif re.search(r"next\s*(?:song|track|music)?", query, re.IGNORECASE):
        last_assistant_question_context_module_level = None; next_song()
    elif re.search(r"resume\s*(?:playing|playback)?|continue playing", query, re.IGNORECASE):
        last_assistant_question_context_module_level = None
        with music_player_lock:
            if player and player.get_state() == vlc.State.Paused: player.play(); is_music_playing_flag = True; cli_speak("Resuming playback.", force_speak=True)
            else: cli_speak("Nothing to resume.", force_speak=True)
    elif (m := re.search(r"(?:weather|forecast)(?:\s+(?:in|for|at|like in)\s+([a-zA-Z\s\-,'.]+))?", query, re.IGNORECASE)): 
        last_assistant_question_context_module_level = None; city = m.group(1).strip() if m.group(1) else None
        if not city: 
            cli_speak("For which city?", force_speak=True)
            city_clarified = cli_command() if _speak_mode == "direct" else None
            if city_clarified and city_clarified != "unintelligible": get_weather_info(city_clarified.strip())
            elif _speak_mode == "capture" and not city_clarified: _streamlit_speak_capture_list.append("(Awaiting city name for weather)")
        else: get_weather_info(city)
    elif "reminder" in query: 
        last_assistant_question_context_module_level = None; 
        if _speak_mode == "direct" or any(kw in query for kw in ["add reminder", "show reminders", "delete reminder", "remove reminder"]):
            manage_reminders_flow()
        else: 
            cli_speak("You can add, view, or delete reminders. For example, say 'add reminder to call John'.")
    elif re.search(r"where\s+am\s+i|my\s+location", query, re.IGNORECASE): last_assistant_question_context_module_level = None; get_location()
    elif re.search(r"system\s+(status|condition|info)|computer health", query, re.IGNORECASE): last_assistant_question_context_module_level = None; check_system_condition()
    elif (m := re.search(r"translate\s*(?:this|the phrase)?\s*['\"]?(.*?)['\"]?\s+to\s+([a-zA-Z\s\-]+)", query, re.IGNORECASE)): 
        last_assistant_question_context_module_level = None; text, lang = (m.group(1).strip() if m.group(1) else ""), m.group(2).strip()
        if not text: 
            cli_speak(f"What to translate to {lang}?", force_speak=True)
            clarified_text = cli_command() if _speak_mode == "direct" else None
            if clarified_text and clarified_text != "unintelligible": translate_text_flow(clarified_text.strip(), lang)
            elif _speak_mode == "capture" and not clarified_text: _streamlit_speak_capture_list.append(f"(Awaiting text to translate to {lang})")
        else: translate_text_flow(text, lang)
    elif (m := re.search(r"open\s+(.+)", query, re.IGNORECASE)):
        last_assistant_question_context_module_level = None; target = m.group(1).strip().lower()
        social = {"facebook":1,"twitter":1,"youtube":1,"instagram":1,"linkedin":1} 
        opened = False
        for s_key in social: 
            if s_key in target: open_social_media(s_key); opened = True; break
        if not opened: open_application(re.sub(r"^(app|program)\s+", "", target, flags=re.I).strip())
    elif (m := re.search(r"close\s+(.+)", query, re.IGNORECASE)): 
        last_assistant_question_context_module_level = None; close_application(m.group(1).strip())
    elif "increase volume" in query or "volume up" in query: last_assistant_question_context_module_level = None; change_system_volume("up")
    elif "decrease volume" in query or "volume down" in query: last_assistant_question_context_module_level = None; change_system_volume("down")
    elif "mute" in query and "volume" in query: last_assistant_question_context_module_level = None; change_system_volume("mute")
    elif "unmute" in query and "volume" in query: last_assistant_question_context_module_level = None; change_system_volume("unmute")
    elif (m := re.search(r"(?:google|search google for)\s+(.+)", query, re.IGNORECASE)):
        last_assistant_question_context_module_level = None; cli_speak(f"Googling '{m.group(1).strip()}'.", force_speak=True); perform_browsing(m.group(1).strip())
    elif (m := re.search(r"(?:detailed search for|scrape|extract about)\s+(.+)", query, re.IGNORECASE)):
        last_assistant_question_context_module_level = None; handle_detailed_web_search(m.group(1).strip())
    elif (m := re.search(r"(?:summarize|tell me about|what is|who is)\s+(.+?)(?:\s+on web)?$", query, re.IGNORECASE)):
        last_assistant_question_context_module_level = None; perform_web_search_and_summarize(m.group(1).strip())
    elif "screenshot" in query or "capture screen" in query:
        last_assistant_question_context_module_level = None
        try:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); ss_dir = os.path.join(script_dir, "screenshots"); os.makedirs(ss_dir, exist_ok=True) 
            fn = os.path.join(ss_dir, f"ss_{ts}.png"); pyautogui.screenshot(fn) 
            cli_speak(f"Screenshot saved in screenshots folder.", force_speak=True)
            try: 
                if sys.platform=="win32": os.startfile(ss_dir)
                elif sys.platform=="darwin": subprocess.run(['open',ss_dir])
                else: subprocess.run(['xdg-open',ss_dir])
            except: pass
        except Exception as e: print(f"Screenshot error: {e}"); cli_speak("Error taking screenshot.", force_speak=True)
    elif re.search(r"time now|current time|what time is it", query, re.IGNORECASE):
        last_assistant_question_context_module_level = None
        current_time_str = datetime.datetime.now().strftime("%I:%M %p").lstrip('0') 
        cli_speak(f"The current time is {current_time_str}.", force_speak=True)
    elif re.search(r"\b(exit|quit|goodbye|bye|terminate)\b", query, re.IGNORECASE): 
        cli_speak_random(["Goodbye!", "See you!", "Shutting down."], force_speak=True); print("Exiting...")
        with music_player_lock: 
            if player:
                if player.is_playing() or player.get_state() == vlc.State.Paused: player.stop()
                player.release(); player = None 
            is_music_playing_flag = False 
        sys.exit(0) 
    else:
        if not handle_general_conversation_query(query): 
            if not is_music_playing_flag: 
                cli_speak(f"I'm not sure about '{query}'.", force_speak=True)
                cli_speak("Shall I search the web?", force_speak=True) 
                last_assistant_question_context_module_level = {'type': 'web_search_confirm', 'query': query}
            else: print(f"[Unknown Suppressed] '{query}'. Music playing."); last_assistant_question_context_module_level = None 
    
    return last_assistant_question_context_module_level


# --- Function for Streamlit to call ---
def get_chatbot_response(query, current_streamlit_context):
    global last_assistant_question_context_module_level, _streamlit_speak_capture_list, model, tokenizer, label_encoder, data
    
    # Ensure NLU models are loaded if they haven't been (e.g., if Streamlit re-imports)
    if model is None or tokenizer is None or label_encoder is None or data is None:
        load_all_models_and_data() # This will attempt to load them
        if model is None: # Still None after attempt
            return "Error: NLU models could not be loaded. Please check the console.", None

    set_speak_mode("capture") 
    _streamlit_speak_capture_list = [] 
    
    last_assistant_question_context_module_level = current_streamlit_context
    
    updated_context_from_logic = process_command_logic(query, current_context=last_assistant_question_context_module_level)
    
    response_text = "\n".join(_streamlit_speak_capture_list)
    
    return response_text, updated_context_from_logic


# --- Main Execution Loop (for CLI Voice Interaction) ---
def run_cli_mode():
    global _speak_mode
    set_speak_mode("direct") 
    load_all_models_and_data() 
    
    if OPENWEATHERMAP_API_KEY == "YOUR_ACTUAL_OPENWEATHERMAP_API_KEY" or not OPENWEATHERMAP_API_KEY or len(OPENWEATHERMAP_API_KEY) < 30:
        print("\n" + "="*60 + "\nWARNING: OPENWEATHERMAP_API_KEY IS NOT SET or is a placeholder!\nWeather functionality WILL FAIL.\nPlease get a key from openweathermap.org and update the script.\n" + "="*60 + "\n")
    
    initialize_cli_tts_engine(); init_db()           
    try:
        if not any(t.name == "ReminderThread" for t in threading.enumerate()): # Avoid duplicate threads
            threading.Thread(target=check_reminders, daemon=True, name="ReminderThread").start()
        if not any(t.name == "VLCMonitorThread" for t in threading.enumerate()):
            threading.Thread(target=vlc_state_monitor, daemon=True, name="VLCMonitorThread").start()
    except Exception as e: print(f"Failed to start background threads in CLI: {e}"); sys.exit(1) 
    
    wishMe(); cli_speak_random(["Ready for commands.", "How can I help?", "Listening."])
    try:
        while True:
            voice_query_raw = cli_command() 
            if not voice_query_raw: time.sleep(0.1); continue
            if voice_query_raw == "unintelligible": cli_speak_random(["Missed that, say again?", "Pardon?"], force_speak=True); last_assistant_question_context_module_level = None; continue
            voice_query = voice_query_raw.strip() 
            if not voice_query: continue
            print(f"User: '{voice_query}'")
            last_assistant_question_context_module_level = process_command_logic(voice_query, current_context=last_assistant_question_context_module_level)
            time.sleep(0.1) 
    except KeyboardInterrupt: print("\nExiting (Ctrl+C).");
    except Exception as e: print(f"\nFATAL ERROR: {e}"); import traceback; traceback.print_exc()
    finally:
        print("Application terminated.");
        with music_player_lock: 
            if player:
                try:
                    if player.is_playing(): player.stop()
                    player.release()
                except: pass
                player = None
        for app_name, proc_obj in list(opened_processes.items()):
            try:
                if proc_obj.poll() is None: print(f"Terminating {app_name} on exit..."); proc_obj.terminate(); proc_obj.wait(timeout=1)
            except: pass

if __name__ == "__main__":
    # This allows running 'python main.py' for CLI
    # or 'streamlit run app_streamlit.py' which will import this main.py
    # If Streamlit imports this, the __main__ block won't run, preventing CLI mode from starting.
    # The Streamlit app (app_streamlit.py) will call get_chatbot_response().
    
    # If you intend to run Streamlit directly from this file, you'd need:
    # if len(sys.argv) > 1 and sys.argv[1] == 'streamlit_app_mode':
    #     run_streamlit_app() # Assuming you define run_streamlit_app() in this file too
    # else:
    #     run_cli_mode()
    
    # For clarity and best practice, keep app_streamlit.py separate and let it import this.
    # This __main__ block is now solely for CLI execution.
    run_cli_mode()
