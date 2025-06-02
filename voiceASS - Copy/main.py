import datetime
import os
import sys
import time
import webbrowser
import pyautogui
import pyttsx3
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
import vlc
import yt_dlp
import requests
try:
    from googletrans import Translator
except ImportError:
    print("googletrans not found, using google_trans_new as fallback. Consider 'pip install googletrans==4.0.0rc1'")
    from google_trans_new import google_translator # Fallback
    
from googlesearch import search
from bs4 import BeautifulSoup
from gtts import gTTS
from playsound import playsound
import tempfile
from deep_translator import GoogleTranslator

# Imports for Reminder System
import sqlite3
import threading
import re

# --- Global Variables ---
player = None
main_recognizer = sr.Recognizer() 
translator_service = Translator()
script_dir = os.path.dirname(os.path.abspath(__file__))

tts_engine = None
tts_engine_lock = threading.Lock()

# <<<--- NEW: State variable for music playback --->>>
is_music_playing_flag = False 
music_player_lock = threading.Lock() # To protect access to player and is_music_playing_flag

# --- Load Model and Data ---
intents_path = os.path.join(script_dir, "intents.json")
try:
    with open(intents_path, encoding="utf-8") as file:
        data = json.load(file)
    
    model_path = os.path.join(script_dir, "chat_model.keras")
    tokenizer_path = os.path.join(script_dir, "tokenizer.pkl")
    label_encoder_path = os.path.join(script_dir, "label_encoder.pkl")

    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(label_encoder_path, "rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)

except FileNotFoundError as e:
    print(f"Error: Critical file not found: {e}. Please ensure model and data files are present in '{script_dir}'.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model or data files: {e}")
    sys.exit(1)

language_map = {
    "english": "en", "french": "fr", "german": "de", "korean": "ko",
    "spanish": "es", "chinese": "zh-cn", "japanese": "ja", "arabic": "ar"
}

# --- Utility for varied speech ---
def speak_random(phrases, wait=False, force_speak=False): # Added force_speak
    speak(random.choice(phrases), wait=wait, force_speak=force_speak)


# --- Text-to-Speech ---
def initialize_engine():
    global tts_engine
    if tts_engine is None:
        try:
            engine = pyttsx3.init("sapi5") # Use "sapi5" for Windows
            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
            
            rate = engine.getProperty('rate')
            engine.setProperty('rate', max(50, rate - 50))
            volume = engine.getProperty('volume')
            engine.setProperty('volume', min(1.0, volume + 0.25))
            tts_engine = engine
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
            tts_engine = None
    return tts_engine

def speak(text, wait=False, force_speak=False): # Added force_speak
    global tts_engine_lock, is_music_playing_flag
    
    # <<<--- NEW: Check music state before speaking --->>>
    if is_music_playing_flag and not force_speak:
        print(f"Music is playing. Suppressed speech: '{text}' (use force_speak=True to override)")
        return 

    engine_instance = initialize_engine()
    if engine_instance:
        with tts_engine_lock:
            try:
                engine_instance.say(text)
                engine_instance.runAndWait()
            except RuntimeError as e_runtime:
                print(f"TTS Runtime Error: {e_runtime}. Reinitializing engine.")
                global tts_engine
                tts_engine = None # Force reinitialization
                # Try re-initializing and speaking again, once.
                engine_instance_retry = initialize_engine()
                if engine_instance_retry:
                    try:
                        engine_instance_retry.say(text)
                        engine_instance_retry.runAndWait()
                    except Exception as e_retry_speak:
                        print(f"TTS Error on retry: {e_retry_speak}")
            except Exception as e:
                print(f"Error during speech: {e}")
        if wait:
            time.sleep(0.5)
    else:
        print(f"TTS SKIPPED (engine not initialized): {text}")


# --- Speech-to-Text ---
def command(recognizer_instance=None, prompt_text=None, force_prompt=False): # Added force_prompt
    current_recognizer = recognizer_instance if recognizer_instance else main_recognizer
    
    if prompt_text:
        speak(prompt_text, force_speak=force_prompt) # Use force_prompt here

    with sr.Microphone() as source:
        try:
            current_recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening...", end="", flush=True)
            current_recognizer.pause_threshold = 0.8
            audio = current_recognizer.listen(source, timeout=6, phrase_time_limit=12) # Slightly longer
            print("\rRecognizing...", end="", flush=True)
            query = current_recognizer.recognize_google(audio, language='en-US')
            print(f"\rUser said: {query}\n")
            return query.lower()
        except sr.WaitTimeoutError:
            print("\rNo speech detected.")
            return None
        except sr.UnknownValueError:
            print("\rCould not understand audio.")
            return "unintelligible"
        except sr.RequestError as e:
            print(f"\rSpeech service request error; {e}")
            # Only speak if not related to music interruption logic
            speak_random(["My speech recognition seems to be offline. Please check your internet.",
                          "I'm having trouble understanding right now due to a connection issue."], force_speak=True)
            return None
        except Exception as e:
            print(f"\rUnexpected error in command(): {e}")
            return None

# --- Specialized Speech Input for Reminder Setup ---
def get_reminder_detail(prompt_message, retries=2):
    detail_recognizer = sr.Recognizer()
    
    # For reminder details, we usually want the prompt to be heard
    response = command(recognizer_instance=detail_recognizer, prompt_text=prompt_message, force_prompt=True) 
    
    for attempt in range(retries):
        if response and response != "unintelligible":
            return response
        elif response == "unintelligible":
            if attempt < retries - 1:
                speak_random(["I didn't quite catch that. Could you say it again?", "Sorry, one more time?"], force_speak=True)
                response = command(recognizer_instance=detail_recognizer) # Listen again without prompt
            else:
                speak("I'm still having trouble understanding. Let's skip this part for now.", force_speak=True)
                return None
        elif response is None: # Timeout
             if attempt < retries - 1:
                speak("I didn't hear anything. Please tell me.", force_speak=True)
                response = command(recognizer_instance=detail_recognizer) # Listen again without prompt
             else:
                speak("No input received. Skipping this detail.", force_speak=True)
                return None
    return None # Should be caught by one of the above returns

# --- Reminder System Functions ---
DATABASE_NAME = os.path.join(script_dir, "reminders.db")
def init_db(): # ... (no changes needed)
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reminders (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 description TEXT,
                 datetime TEXT,
                 repeat TEXT)''')
    conn.commit()
    conn.close()

def add_reminder_db(desc, dt_string, repeat_type): # ... (no changes needed)
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO reminders (description, datetime, repeat) VALUES (?, ?, ?)",
              (desc, dt_string, repeat_type))
    conn.commit()
    conn.close()

def remove_reminder_db(reminder_id): # ... (no changes needed)
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
    conn.commit()
    conn.close()

def parse_month(text): # ... (no changes needed)
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
    }
    text = text.lower()
    for name, num in months.items():
        if name in text: return num
    match_num = re.search(r'\b([1-9]|1[0-2])\b', text)
    if match_num: return int(match_num.group(1))
    return None

def listen_in_thread(prompt_text): # For reminder alerts
    thread_recognizer = sr.Recognizer()
    # Critical reminder alerts should always be spoken and listened to
    return command(recognizer_instance=thread_recognizer, prompt_text=prompt_text, force_prompt=True)

# <<<--- NEW: Thread to monitor VLC player state --->>>
def vlc_state_monitor():
    global player, is_music_playing_flag, music_player_lock
    while True:
        with music_player_lock:
            if player:
                current_state = player.get_state()
                # States indicating not playing: Ended, NothingSpecial, Error, Stopped
                if current_state in [vlc.State.Ended, vlc.State.NothingSpecial, vlc.State.Error, vlc.State.Stopped]:
                    if is_music_playing_flag:
                        print("[VLC Monitor] Music stopped or ended.")
                        is_music_playing_flag = False
                        # player.release() # Optional: release player instance if stopped for a while
                        # player = None
                elif current_state == vlc.State.Playing:
                    if not is_music_playing_flag: #  If it was marked false but is playing (e.g. manual restart)
                        is_music_playing_flag = True

            elif is_music_playing_flag: # Player is None but flag is True (e.g., after stop_song)
                is_music_playing_flag = False
        
        time.sleep(1) # Check every second


def check_reminders(): # ... (Reminder logic - mostly fine, but ensure speak calls are forced if necessary)
    global script_dir, is_music_playing_flag
    alert_sound_path = os.path.join(script_dir, "alert.mp3")

    while True:
        now = datetime.datetime.now()
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()
        try:
            c.execute("SELECT * FROM reminders")
            reminders_list = c.fetchall()
        except sqlite3.Error as e:
            print(f"Database error in check_reminders: {e}")
            time.sleep(60) 
            if conn: conn.close() # Ensure close on error before continue
            continue
        finally:
            if conn: conn.close()

        for reminder_id, desc, dt_str, repeat_type in reminders_list:
            try:
                reminder_dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
            except ValueError:
                print(f"Skipping reminder {reminder_id} due to invalid datetime string: {dt_str}")
                remove_reminder_db(reminder_id)
                print(f"Removed malformed reminder {reminder_id}.")
                continue

            alert = False
            # ... (alert condition logic) ...
            current_day_name = now.strftime("%A").lower()
            current_time_hm = now.strftime("%H:%M")

            if repeat_type.startswith("weekly-"):
                reminder_weekday = repeat_type.split("-", 1)[1]
                if reminder_weekday == current_day_name and \
                   reminder_dt.strftime("%H:%M") == current_time_hm and \
                   now.date() >= reminder_dt.date(): 
                    alert = True
            elif repeat_type == "once":
                if now >= reminder_dt: 
                    alert = True
            
            if alert:
                print(f"ALERT: Reminder ID {reminder_id} - {desc}")
                
                # Temporarily pause music if playing, for the reminder
                music_was_paused_for_reminder = False
                with music_player_lock:
                    if is_music_playing_flag and player and player.is_playing():
                        player.pause()
                        music_was_paused_for_reminder = True
                        print("[Reminder] Paused music for reminder.")
                        time.sleep(0.5) # Give a moment for music to actually pause

                if os.path.exists(alert_sound_path):
                    try:
                        playsound(alert_sound_path, block=False)
                    except Exception as e_ps:
                        print(f"Error playing alert sound: {e_ps}")
                        speak("Reminder:", force_speak=True) 
                else:
                    speak("Reminder:", force_speak=True)

                speak(f"{desc}", force_speak=True)

                if repeat_type == "once":
                    remove_reminder_db(reminder_id)
                    speak("This was a one-time reminder and has now been removed.", force_speak=True)
                elif repeat_type.startswith("weekly-"):
                    response = listen_in_thread("This is a weekly reminder. Should I remove it, or keep it active?") # listen_in_thread forces prompt
                    if response:
                        if "remove" in response or "delete" in response or "stop it" in response:
                            remove_reminder_db(reminder_id)
                            speak("Okay, I've removed this weekly reminder.", force_speak=True)
                        else:
                            speak("Alright, I'll keep it for next week.", force_speak=True)
                    else:
                        speak("I'll keep the weekly reminder active.", force_speak=True)
                
                # Resume music if it was paused for the reminder
                if music_was_paused_for_reminder:
                    with music_player_lock:
                        if player and not player.is_playing(): # Check if it's still in a pausable state
                             # Check player state before trying to play. It might have been stopped by user.
                            player_state_before_resume = player.get_state()
                            if player_state_before_resume == vlc.State.Paused:
                                player.play() # In VLC, play() resumes from pause
                                print("[Reminder] Resumed music after reminder.")
                            else:
                                print(f"[Reminder] Music state changed to {player_state_before_resume}, not resuming automatically.")
                        # else: player might have been stopped/released by another command

        time.sleep(25)

def manage_reminders_flow(): # ... (ensure speak calls inside use force_speak=True for prompts)
    speak_random(["Sure, I can help with that. Do you want a weekly reminder or a one-time reminder?",
                  "Setting a reminder! Will this be a weekly one, or just for a single occasion?"], force_speak=True)
    choice_text = command(force_prompt=False) # Prompt already spoken

    if not choice_text or choice_text == "unintelligible":
        speak("I didn't quite catch the reminder type. We can try again later.", force_speak=True)
        return
    # ... (rest of the logic, make sure all speak() for prompts have force_speak=True)
    reminder_type = ""
    target_day_name = None

    if "weekly" in choice_text:
        reminder_type = "weekly"
        valid_days = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
        while True:
            day_name_input = get_reminder_detail("Which day of the week for this weekly reminder?") # get_reminder_detail forces its prompt
            if day_name_input and day_name_input != "unintelligible":
                matched_day = None
                for day_val in valid_days:
                    if day_val in day_name_input:
                        matched_day = day_val; break
                if matched_day:
                    target_day_name = matched_day; break
                else:
                    speak_random(["Hmm, I don't recognize that day. Please say a day like Monday, Tuesday, etc.",
                                  "Could you repeat the day of the week?"], force_speak=True)
            elif day_name_input is None:
                speak("Looks like we missed the day. Let's cancel this reminder setup for now.", force_speak=True); return
            if day_name_input and "cancel" in day_name_input: speak("Okay, cancelling reminder setup.", force_speak=True); return
        speak(f"Got it, a weekly reminder for every {target_day_name}.", force_speak=True)

    elif "one time" in choice_text or "once" in choice_text or "single" in choice_text:
        reminder_type = "once"
        speak("Alright, a one-time reminder.", force_speak=True)
    else:
        speak("I'm not sure about that type. If you want to set a reminder, please tell me if it's weekly or one-time.", force_speak=True)
        return

    now_dt = datetime.datetime.now()
    year_text = get_reminder_detail(f"What year? For example, {now_dt.year} or {now_dt.year + 1}.")
    if not year_text or not year_text.isdigit(): speak("Invalid year. Cancelling.", force_speak=True); return
    year = int(year_text)

    month_text = get_reminder_detail("Which month? You can say the name or number.")
    month = parse_month(month_text)
    if not month: speak("Invalid month. Cancelling.", force_speak=True); return
    
    day_text = get_reminder_detail("And what day of the month?")
    if not day_text or not day_text.isdigit(): speak("Invalid day. Cancelling.", force_speak=True); return
    day = int(day_text)

    hour_text = get_reminder_detail("What hour, using 24-hour format? For instance, 14 for 2 PM.")
    if not hour_text or not hour_text.isdigit() or not (0 <= int(hour_text) <= 23): speak("Invalid hour. Cancelling.", force_speak=True); return
    hour = int(hour_text)

    minute_text = get_reminder_detail("And the minute?")
    if not minute_text or not minute_text.isdigit() or not (0 <= int(minute_text) <= 59): speak("Invalid minute. Cancelling.", force_speak=True); return
    minute = int(minute_text)

    try:
        dt_obj = datetime.datetime(year, month, day, hour, minute)
        if dt_obj < now_dt and reminder_type == "once":
            speak("This date and time is in the past for a one-time reminder. Please provide a future time.", force_speak=True)
            return 
    except ValueError:
        speak("The date or time you provided doesn't seem right. Let's try setting the reminder again later.", force_speak=True)
        return

    speak(f"So, that's a reminder for {dt_obj.strftime('%A, %B %d, %Y at %I:%M %p')}. Is this correct?", force_speak=True)
    confirmation = command(force_prompt=False)

    if confirmation and ("yes" in confirmation or "correct" in confirmation or "yeah" in confirmation or "okay" in confirmation or "right" in confirmation):
        desc = get_reminder_detail("Great! What should I remind you about?")
        if not desc or desc == "unintelligible":
            speak("I didn't get the description for the reminder, so I can't save it.", force_speak=True)
            return
        
        final_repeat_type = f"weekly-{target_day_name}" if reminder_type == "weekly" and target_day_name else "once"
        add_reminder_db(desc, dt_obj.strftime("%Y-%m-%d %H:%M"), final_repeat_type)
        speak_random(["Reminder saved!", "All set. I'll remind you.", "Okay, I've noted that down."], force_speak=True)
    else:
        speak_random(["Okay, I've cancelled this reminder.", "No problem, reminder discarded."], force_speak=True)

    more_q = command(prompt_text="Would you like to set another reminder?", force_prompt=True)
    if more_q and ("yes" in more_q or "sure" in more_q):
        manage_reminders_flow()
    else:
        speak("Alright.", force_speak=True) # Confirm end of flow if music is not playing
# --- Main Assistant's Functions ---

def speak_in_language(text, lang_code): # Should generally be forced
    try:
        tts = gTTS(text=text, lang=lang_code)
        temp_file_path = os.path.join(script_dir, f"temp_tts_{lang_code}.mp3")
        tts.save(temp_file_path)
        playsound(temp_file_path, block=True)
        os.remove(temp_file_path)
    except Exception as e:
        print(f"Error in speak_in_language: {e}")
        speak_random(["Sorry, I had trouble speaking in that language.", "My apologies, I couldn't vocalize the translation."], force_speak=True)

def get_location(): # User explicitly asked, so force_speak for the result
    speak_random(["Let me check your current location...", "Figuring out where you are..."], force_speak=True)
    try:
        response = requests.get("https://ipinfo.io/json", timeout=7)
        response.raise_for_status()
        data = response.json()
        city = data.get('city', 'an unknown city')
        region = data.get('region', 'an unknown region')
        country = data.get('country', 'an unknown country')
        speak(f"It looks like you're in {city}, {region}, in {country}.", force_speak=True)
        return True
    except requests.exceptions.RequestException as e:
        speak("I'm having trouble reaching the location service. Please check your internet connection.", force_speak=True)
        print(f"Location error: {e}")
        return False
    except Exception as e:
        speak("An unexpected issue occurred while trying to find your location.", force_speak=True)
        print(f"Unexpected location error: {e}")
        return False

def search_youtube(query_song): # ... (no changes to internal logic, but caller handles speak)
    ydl_opts = {'quiet': True, 'skip_download': True, 'extract_flat': 'in_playlist', 'default_search': 'ytsearch1'}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query_song}", download=False)
            if 'entries' in info and info['entries']:
                video_id = info['entries'][0]['id']
                video_title = info['entries'][0].get('title', query_song)
                return f"https://www.youtube.com/watch?v={video_id}", video_title
            elif 'id' in info:
                 video_id = info['id']
                 video_title = info.get('title', query_song)
                 return f"https://www.youtube.com/watch?v={video_id}", video_title
    except Exception as e:
        print(f"YouTube search error for '{query_song}': {e}")
    return None, None


def get_best_audio_url(video_url): # ... (no changes)
    ydl_opts = {'format': 'bestaudio/best', 'quiet': True, 'noplaylist': True, 'extract_flat': False}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            return info_dict.get('url')
    except Exception as e:
        print(f"Error getting audio URL from '{video_url}': {e}")
    return None


def play_song(song_query):
    global player, is_music_playing_flag, music_player_lock
    # This is a direct command, so initial speak is fine (forced)
    speak_random([f"Looking for {song_query} on YouTube...", f"Let me find {song_query} for you."], force_speak=True)
    
    with music_player_lock: # Ensure thread-safe access to player and flag
        try:
            if player and player.is_playing(): 
                player.stop()
                is_music_playing_flag = False # Explicitly set here
            
            video_url, video_title = search_youtube(song_query)
            if not video_url:
                speak(f"Sorry, I couldn't find '{song_query}' on YouTube.", force_speak=True)
                return
                
            playurl = get_best_audio_url(video_url)
            if not playurl:
                speak(f"I found '{video_title}', but I'm having trouble getting an audio stream for it.", force_speak=True)
                return

            try:
                instance = vlc.Instance('--no-xlib --quiet --verbose=-1')
                player = instance.media_player_new()
            except Exception as e_vlc_init:
                print(f"VLC Initialization error: {e_vlc_init}")
                speak("I'm having a problem with my audio player setup.", force_speak=True)
                return

            media = instance.media_new(playurl)
            player.set_media(media)
            player.play()
            
            # Wait a moment for VLC to actually start playing
            time.sleep(1) # Check state after a brief pause
            if player.get_state() == vlc.State.Playing:
                is_music_playing_flag = True
                print(f"[Music] Now playing: {video_title}")
                speak(f"Now playing: {video_title}.", force_speak=True) # Announce the song
            else:
                is_music_playing_flag = False # Failed to play
                print(f"[Music] Failed to start playback for: {video_title}. State: {player.get_state()}")
                speak(f"I tried to play {video_title}, but it didn't start correctly.", force_speak=True)


        except Exception as e:
            print(f"Generic playback error for '{song_query}': {e}")
            is_music_playing_flag = False # Ensure flag is false on error
            speak_random(["Oops, something went wrong while trying to play the song.",
                        "I encountered an error with music playback."], force_speak=True)

def stop_song():
    global player, is_music_playing_flag, music_player_lock
    with music_player_lock:
        if player and player.is_playing(): # Check if it was actually playing
            player.stop()
            is_music_playing_flag = False
            speak_random(["Music stopped.", "Playback halted."], force_speak=True)
            # player.release() # Consider releasing the player instance
            # player = None 
        elif player and not player.is_playing() and is_music_playing_flag: # Was flagged as playing but VLC says no (e.g. paused)
            player.stop() # Ensure it's fully stopped
            is_music_playing_flag = False
            speak_random(["Okay, music stopped.", "Playback ended."], force_speak=True)
        else: # No active player or not playing
            is_music_playing_flag = False # Ensure flag consistency
            speak_random(["There's no music playing right now.", "Nothing to stop."], force_speak=True)


def cal_day_name(): # ... (no changes)
    today_weekday = datetime.datetime.today().weekday()
    day_dict = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    return day_dict.get(today_weekday, "an unknown day")

def wishMe(): # Initial greeting, always spoken
    hour = int(datetime.datetime.now().hour)
    current_time_str = time.strftime("%I:%M %p")
    current_day_str = cal_day_name()

    greeting = "Hello Abdallah"
    if 0 <= hour < 12: greeting = f"Good morning Abdallah"
    elif 12 <= hour < 17: greeting = f"Good afternoon Abdallah"
    else: greeting = f"Good evening Abdallah"
    
    speak(f"{greeting}. It's {current_day_str}, and the time is currently {current_time_str}.", force_speak=True)

def handle_social_media(query): # User explicitly asked
    sites = {
        'facebook': "https://www.facebook.com/", 'whatsapp': "https://web.whatsapp.com/",
        'discord': "https://discord.com/", 'instagram': "https://www.instagram.com/"
    }
    for site_name, url in sites.items():
        if site_name in query:
            speak_random([f"Opening {site_name} for you.", f"Sure, heading to {site_name}."], force_speak=True)
            webbrowser.open(url)
            return True
    return False


def tell_schedule(): # User explicitly asked
    day_today = cal_day_name()
    speak_random([f"Let's see what's on your schedule for today, {day_today}.",
                  f"Checking your schedule for {day_today}."], force_speak=True)
    
    week_schedule = { # Your specific schedule
        "Monday": "You have Algorithms class from 9:00 to 9:50, then System Design from 10:00 to 11:50. After a break, Programming Lab starts from 2:00 PM onwards.",
        "Tuesday": "Web Development class is from 9:00 to 9:50, followed by a break. Then Database Systems class from 11:00 to 12:50. After another break, Open Source Projects lab begins at 2:00 PM.",
        "Wednesday": "It's a full day. Machine Learning class from 9:00 to 10:50, Operating Systems from 11:00 to 11:50, and Ethics in Technology from 12:00 to 12:50. After a break, Software Engineering workshop is from 2:00 PM.",
        "Thursday": "Another busy day. Computer Networks class from 9:00 to 10:50, then Cloud Computing from 11:00 to 12:50. After a break, Cybersecurity lab is from 2:00 PM.",
        "Friday": "Artificial Intelligence class from 9:00 to 9:50, Advanced Programming from 10:00 to 10:50, and UI/UX Design from 11:00 to 12:50. After a break, Capstone Project work is from 2:00 PM.",
        "Saturday": "A more relaxed day. Team meetings for your Capstone Project from 9:00 to 11:50, Innovation and Entrepreneurship class from 12:00 to 12:50. The afternoon is for personal development and coding practice from 2:00 PM.",
        "Sunday": "It's Sunday, a holiday! But remember to keep an eye on upcoming deadlines and maybe catch up on some reading or project work."
    }
    
    if day_today in week_schedule:
        speak(week_schedule[day_today], force_speak=True)
    else:
        speak("Hmm, I seem to be having trouble finding the schedule for today.", force_speak=True)

APP_CONFIG = { # ... (no changes to config itself)
    "calculator": {'open_cmd': 'calc.exe', 'process_name': ["Calculator.exe", "CalculatorApp.exe"], 'type': 'system_exe'},
    "notepad": {'open_cmd': 'notepad.exe', 'process_name': ["notepad.exe"], 'type': 'system_exe'},
    "paint": {'open_cmd': 'mspaint.exe', 'process_name': ["mspaint.exe"], 'type': 'system_exe'},
    "visual studio code": {'open_cmd': 'code', 'process_name': ["Code.exe"], 'type': 'path_command'},
    "vs code": {'open_cmd': 'code', 'process_name': ["Code.exe"], 'type': 'path_command'}
}

def manage_app(action, app_name_query): # User explicitly asked
    app_to_manage = None
    sorted_app_keys = sorted(APP_CONFIG.keys(), key=len, reverse=True)
    for key in sorted_app_keys:
        if key in app_name_query:
            app_to_manage = key; break
    
    if not app_to_manage:
        speak(f"I'm not sure which application you're referring to for the '{action}' command.", force_speak=True)
        return

    config = APP_CONFIG[app_to_manage]
    if action == "open":
        speak_random([f"Sure, opening {app_to_manage}.", f"Alright, launching {app_to_manage}."], force_speak=True)
        try:
            if config['type'] == 'system_exe': os.startfile(config['open_cmd'])
            elif config['type'] == 'path_command': subprocess.Popen([config['open_cmd']])
        except FileNotFoundError:
            speak(f"Sorry, I couldn't find {app_to_manage}. It might not be installed or in the system path.", force_speak=True)
        except Exception as e:
            speak(f"I ran into an issue trying to open {app_to_manage}.", force_speak=True)
            print(f"Error opening {app_to_manage} ({config['open_cmd']}): {e}")
    elif action == "close":
        speak_random([f"Attempting to close {app_to_manage}.", f"Alright, let's try to close {app_to_manage}."], force_speak=True)
        killed = False
        for proc_name_variant in config['process_name']:
            try:
                for proc in psutil.process_iter(['pid', 'name']):
                    if proc.info['name'].lower() == proc_name_variant.lower():
                        proc.terminate()
                        try: proc.wait(timeout=2)
                        except psutil.TimeoutExpired: print(f"{proc_name_variant} did not terminate gracefully, killing."); proc.kill()
                        speak(f"{app_to_manage} should be closed now.", force_speak=True)
                        killed = True; break 
                if killed: break
            except psutil.NoSuchProcess: continue
            except Exception as e: print(f"Error during process termination for {proc_name_variant}: {e}")
        if not killed:
            speak_random([f"I couldn't find {app_to_manage} running, or it was already closed.",
                            f"{app_to_manage} doesn't seem to be open."], force_speak=True)


def handle_browsing(query): # User explicitly asked, so prompts/results are forced
    search_google_match = re.search(r"(?:search google for|google search|google)\s+(.+)", query, re.IGNORECASE)
    open_edge_match = re.search(r"open\s+(?:microsoft\s)?edge", query, re.IGNORECASE)

    if search_google_match:
        search_term = search_google_match.group(1).strip()
        if search_term:
            speak_random([f"Sure, searching Google for {search_term}.", f"Googling {search_term} for you."], force_speak=True)
            webbrowser.open(f"https://www.google.com/search?q={search_term}")
        else:
            speak("What would you like me to search on Google?", force_speak=True)
            term_from_user = command(force_prompt=False) # Prompt already spoken
            if term_from_user and term_from_user != "unintelligible":
                speak(f"Okay, searching Google for {term_from_user}.", force_speak=True)
                webbrowser.open(f"https://www.google.com/search?q={term_from_user}")
        return True
    elif open_edge_match:
        speak_random(["Opening Microsoft Edge.", "Launching Edge browser."], force_speak=True)
        try: subprocess.Popen(['start', 'msedge'], shell=True) 
        except Exception as e:
            speak("I had trouble opening Microsoft Edge. Make sure it's installed.", force_speak=True)
            print(f"Error opening Edge: {e}")
        return True
    return False

def report_system_condition(): # User explicitly asked
    speak_random(["Let me check the system's condition.", "Checking system status..."], force_speak=True)
    cpu_usage = psutil.cpu_percent(interval=0.5)
    speak(f"The CPU is currently at {cpu_usage:.1f} percent utilization.", force_speak=True)
    try:
        battery = psutil.sensors_battery()
        if battery:
            percentage = battery.percent
            speak(f"System battery is at {percentage} percent.", force_speak=True)
            if battery.power_plugged: speak("And it's currently plugged in and charging.", force_speak=True)
            else: speak("And it's running on battery power.", force_speak=True)
            if percentage < 20: speak("The battery is quite low, you might want to plug it in soon.", force_speak=True)
            elif percentage < 50: speak("Battery level is moderate.", force_speak=True) # Can be optional if music playing
        else: speak("I couldn't retrieve battery information for this system.", force_speak=True)
    except AttributeError: speak("Battery information doesn't seem applicable for this device.", force_speak=True)
    except Exception as e:
        speak("An error occurred while checking battery status.", force_speak=True)
        print(f"Battery check error: {e}")

def translate_text_with_google(text, target_lang_code, source_lang_code='auto'): # Internal helper, speak handled by caller
    try:
        if isinstance(translator_service, Translator):
            translated = translator_service.translate(text, dest=target_lang_code, src=source_lang_code)
            return translated.text
        else:
             print("Using deep_translator as fallback for translation.")
             return GoogleTranslator(source=source_lang_code, target=target_lang_code).translate(text)
    except Exception as e:
        print(f"Translation error with {type(translator_service).__name__}: {e}")
        # Caller should handle speaking the error.
        return None


def perform_web_search_and_summarize(query_text): # User explicitly asked, all speech forced
    speak_random([f"Let me look that up for you: {query_text}", f"Searching the web for {query_text}."], force_speak=True)
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        search_results = list(search(query_text, num_results=1, lang='en', stop=1, user_agent=headers.get('User-Agent')))
    except Exception as e_search:
        print(f"Googlesearch error: {e_search}")
        speak("I had an issue with the web search. My apologies.", force_speak=True)
        return

    if not search_results:
        speak("I couldn't find any immediate results for that query.", force_speak=True)
        return
    
    first_url = search_results[0]
    print(f"Top result URL: {first_url}")
    speak_random(["I found a page that might be helpful. Let me try to get a quick summary.",
                  "Okay, got a result. I'll try to summarize it for you."], force_speak=True)
    
    page_content_summary = None
    try:
        response = requests.get(first_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element_type in ['script', 'style', 'nav', 'footer', 'aside', 'header', 'form', 'button', 'img', 'iframe', 'link', 'meta']:
            for element in soup.find_all(element_type): element.decompose()
        paragraphs = soup.find_all('p')
        text_parts = [p.get_text(separator=' ', strip=True) for p in paragraphs if p.get_text(strip=True)]
        if text_parts:
            full_text = ' '.join(text_parts)
            sentences = re.split(r'(?<=[.!?])\s+', full_text)
            summary_parts = []
            word_count = 0
            for sentence in sentences:
                summary_parts.append(sentence)
                word_count += len(sentence.split())
                if word_count > 60 : break
            page_content_summary = ' '.join(summary_parts)
    except requests.exceptions.RequestException as e_req:
        print(f"Error fetching page content from {first_url}: {e_req}")
        speak("I could reach the page, but had trouble getting its content for a summary.", force_speak=True)
    except Exception as e_parse:
        print(f"Error parsing page content from {first_url}: {e_parse}")
        speak("I found the page, but encountered an issue while trying to summarize it.", force_speak=True)

    if page_content_summary:
        print(f"Summary: {page_content_summary}\n")
        speak(f"Here's a brief summary from the page: {page_content_summary}", force_speak=True)
    else:
        speak(f"I couldn't extract a concise summary from the page. The top result is at {first_url.split('//')[1].split('/')[0]}.", force_speak=True)

    open_page_q = command(prompt_text="Would you like me to open this page for you to read more?", force_prompt=True)
    if open_page_q and ("yes" in open_page_q or "sure" in open_page_q or "okay" in open_page_q or "please" in open_page_q):
        webbrowser.open(first_url)
        speak_random(["Opening the page now.", "Alright, bringing up the page."], force_speak=True)


def predict_intent_from_text(text): # ... (no changes)
    try:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=20, padding='post', truncating='post')
        prediction = model.predict(padded, verbose=0)
        tag_index = np.argmax(prediction)
        tag = label_encoder.inverse_transform([tag_index])[0]
        confidence = float(prediction[0][tag_index])
        return tag, confidence
    except Exception as e:
        print(f"Error in predict_intent_from_text: {e}")
        return None, 0.0

def handle_general_conversation_query(query_text): # For chitchat, respect music state
    tag, confidence = predict_intent_from_text(query_text)
    print(f"Intent Prediction -> Tag: {tag}, Confidence: {confidence:.3f}")
    CONFIDENCE_THRESHOLD = 0.70

    if tag and confidence > CONFIDENCE_THRESHOLD:
        for intent_data in data['intents']:
            if intent_data['tag'] == tag:
                response_text = random.choice(intent_data['responses'])
                # For general chit-chat, if music is playing, maybe don't speak, or make it brief.
                # If it's a greeting like "hello", responding is fine.
                # If it's "how are you", responding is also fine.
                # The `speak` function will handle the `is_music_playing_flag` check.
                # For critical intents like "goodbye", they are handled directly.
                if tag in ["greetings", "about_ai", "creator", "what_can_you_do", "thanks"]: # Tags that are okay to speak over music
                    speak(response_text, force_speak=True) # Force speak for these specific conversational tags
                else:
                    speak(response_text) # Let default music check happen
                return True 
    return False

# --- Main Execution Loop ---
if __name__ == "__main__":
    initialize_engine()
    init_db()
    
    reminder_thread = threading.Thread(target=check_reminders, daemon=True)
    reminder_thread.start()

    # <<<--- NEW: Start VLC state monitor thread --->>>
    vlc_monitor_thread = threading.Thread(target=vlc_state_monitor, daemon=True)
    vlc_monitor_thread.start()
    
    wishMe() # Initial greeting, always forced
    
    # Only say "I'm ready" if no music is playing initially (unlikely at startup, but good practice)
    speak_random(["I'm ready for your commands.", "How can I help you today?", "Listening."])

    while True:
        query = command(force_prompt=False) # Don't force a prompt for general listening loop
        
        if not query:
            time.sleep(0.1)
            continue
        
        if query == "unintelligible":
            speak_random(["Sorry, I missed that. Could you say it again?", 
                          "I didn't quite catch that. What was it?",
                          "Pardon?"], force_speak=True) # Important to hear this clarification
            continue
            
        print(f"User Command: '{query}'")
        
        # --- Direct Command Handling ---
        play_match = re.search(r"(?:play|listen to|put on|i want to hear)\s+(?:the\s+)?(?:song\s+|music\s+|track\s+)?(.+)", query, re.IGNORECASE)
        if play_match:
            song_name_to_play = play_match.group(1).strip()
            if song_name_to_play: play_song(song_name_to_play)
            else:
                speak("Sure, what song would you like me to play?", force_speak=True)
                song_clarified = command(force_prompt=False)
                if song_clarified and song_clarified != "unintelligible": play_song(song_clarified)
            continue

        elif re.search(r"stop\s+(?:the\s+)?(song|music|playback|playing)", query, re.IGNORECASE) or "stop playing" in query:
            stop_song()
            continue
        
        # Pause/Resume Music (NEW)
        elif re.search(r"pause\s+(?:the\s+)?(song|music|playback)", query, re.IGNORECASE):
            with music_player_lock:
                if player and player.is_playing():
                    player.pause()
                    is_music_playing_flag = False # Paused is not actively "playing" for suppression logic
                    speak("Music paused.", force_speak=True)
                elif player and not player.is_playing() and not is_music_playing_flag: # Already paused
                     speak("Music is already paused.", force_speak=True)
                else:
                    speak("There's no music playing to pause.", force_speak=True)
            continue
        elif re.search(r"resume\s+(?:the\s+)?(song|music|playback)|play music", query, re.IGNORECASE) and \
             not play_match: # Avoid conflict with "play new_song"
            with music_player_lock:
                if player and not player.is_playing(): # Check if player exists and is pausable/stoppable
                    current_vlc_state = player.get_state()
                    if current_vlc_state == vlc.State.Paused:
                        player.play() # play() resumes from Paused state in VLC
                        is_music_playing_flag = True
                        speak("Resuming music.", force_speak=True)
                    elif current_vlc_state in [vlc.State.Stopped, vlc.State.Ended] and player.get_media():
                        # If stopped but media is still set, try playing again (like restart)
                        player.play()
                        is_music_playing_flag = True
                        speak("Restarting the track.", force_speak=True)
                    else:
                         speak("There's no paused music to resume, or the previous track ended.", force_speak=True)
                else:
                    speak("Nothing to resume, or music is already playing.", force_speak=True)
            continue

        elif re.search(r"(?:set|add|new|create)\s+(?:a\s+)?reminder", query, re.IGNORECASE):
            manage_reminders_flow()
            continue

        elif re.search(r"open\s+(facebook|whatsapp|discord|instagram)", query, re.IGNORECASE) or \
             any(site in query for site in ['facebook', 'whatsapp', 'discord', 'instagram']):
            if handle_social_media(query): continue
        
        elif re.search(r"(?:what's|show me|tell me about)\s+(?:my\s+)?(?:university\s+)?(schedule|time\s*table)|my day", query, re.IGNORECASE):
            tell_schedule()
            continue

        elif re.search(r"(volume\s+up|increase\s+volume|turn\s+it\s+up)", query, re.IGNORECASE):
            pyautogui.press("volumeup"); speak("Volume increased.", force_speak=True)
            continue
        elif re.search(r"(volume\s+down|decrease\s+volume|turn\s+it\s+down)", query, re.IGNORECASE):
            pyautogui.press("volumedown"); speak("Volume decreased.", force_speak=True)
            continue
        elif re.search(r"(mute|unmute|silence)\s+(?:the\s+)?(?:volume|sound)?", query, re.IGNORECASE):
            pyautogui.press("volumemute"); speak("Volume mute toggled.", force_speak=True)
            continue
        
        open_app_match = re.search(r"open\s+(?:the\s+)?(.+)", query, re.IGNORECASE)
        close_app_match = re.search(r"close\s+(?:the\s+)?(.+)", query, re.IGNORECASE)
        if open_app_match:
            app_name_part = open_app_match.group(1).strip()
            if not (handle_browsing(f"open {app_name_part}") or handle_social_media(f"open {app_name_part}")):
                 manage_app("open", app_name_part)
            continue
        elif close_app_match:
            manage_app("close", close_app_match.group(1).strip())
            continue
        
        if handle_browsing(query): continue

        elif re.search(r"(?:system|pc|computer)\s+(condition|status|health)|how is the system", query, re.IGNORECASE):
            report_system_condition()
            continue

        elif re.search(r"where\s+am\s+i|what's?\s+my\s+location", query, re.IGNORECASE):
            get_location()
            continue

        if "translate" in query:
             speak("Okay, what text should I translate?", force_speak=True)
             text_to_translate = command(force_prompt=False)
             if text_to_translate and text_to_translate != "unintelligible":
                 speak("And to which language should I translate it? For example, French, Spanish, German.", force_speak=True)
                 lang_name_input = command(force_prompt=False)
                 if lang_name_input and lang_name_input != "unintelligible":
                     target_lang_code = language_map.get(lang_name_input.lower().strip())
                     if target_lang_code:
                         translated_text = translate_text_with_google(text_to_translate, target_lang_code)
                         if translated_text:
                             print(f"Translated to {lang_name_input} ({target_lang_code}): {translated_text}")
                             speak(f"The translation to {lang_name_input} is: {translated_text}", force_speak=True)
                             hear_it_q = command(prompt_text=f"Would you like me to say that in {lang_name_input}?", force_prompt=True)
                             if hear_it_q and ("yes" in hear_it_q or "sure" in hear_it_q):
                                 speak_in_language(translated_text, target_lang_code)
                     else:
                         speak(f"I don't have {lang_name_input} in my list of supported languages for translation, or I misheard.", force_speak=True)
                 else: speak("I didn't catch the target language.", force_speak=True)
             else: speak("I didn't catch the text to translate.", force_speak=True)
             continue

        search_web_match = re.search(r"(?:search\s+(?:for|the web for|web for)|look\s+up|find\s+information\s+(?:about|on))\s+(.+)", query, re.IGNORECASE)
        if search_web_match:
            search_query_text = search_web_match.group(1).strip()
            if search_query_text:
                perform_web_search_and_summarize(search_query_text)
            else:
                speak("Sure, what would you like me to search the web for?", force_speak=True)
                term_clarified = command(force_prompt=False)
                if term_clarified and term_clarified != "unintelligible": perform_web_search_and_summarize(term_clarified)
            continue
        
        elif re.search(r"exit|quit|goodbye|see\s+you|turn\s+off|shut\s+down", query, re.IGNORECASE):
            speak_random(["Goodbye Abdallah! Have a great day.", "See you later, Abdallah!", "Shutting down. Take care!"], force_speak=True)
            with music_player_lock:
                if player and player.is_playing(): player.stop()
                is_music_playing_flag = False # Ensure flag is clear on exit
            sys.exit()
            
        else:
            if not handle_general_conversation_query(query):
                # If music is playing, don't give a confused response unless user seems to be trying hard.
                # For now, just let it pass if music is playing and it's not understood.
                # If music is NOT playing, then give the confused response.
                if not is_music_playing_flag:
                    speak_random([
                        "I'm not quite sure how to help with that. Could you try rephrasing?",
                        "Hmm, I didn't understand that. Would you like me to search the web for it?",
                    ], force_speak=True) # Force speak this confusion if no music
                    
                    # Only ask to search web if music isn't playing
                    if not is_music_playing_flag:
                        clarification_q = command(prompt_text="Shall I search the web for that?", force_prompt=True)
                        if clarification_q and ("yes" in clarification_q or "sure" in clarification_q):
                             perform_web_search_and_summarize(query)
