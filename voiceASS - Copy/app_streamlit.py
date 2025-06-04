import streamlit as st
import main as chatbot_logic # ملفك الرئيسي الآن
import time
from gtts import gTTS
import os
import tempfile
from PIL import Image 
import datetime 
import random 
import re 
import threading

# --- تصميم الواجهة (يجب أن يكون أول شيء) ---
st.set_page_config(page_title="Abdallah AI Assistant", layout="wide", page_icon="🤖")

# --- تهيئة الشات بوت عند بدء التطبيق (مرة واحدة) ---
@st.cache_resource 
def initialize_chatbot_backend_streamlit():
    print("Streamlit: Initializing chatbot backend...")
    chatbot_logic.load_all_models_and_data() 
    chatbot_logic.init_db()
    
    thread_names = [t.name for t in threading.enumerate()]
    if "ReminderThreadStreamlit" not in thread_names:
        print("Streamlit: Starting ReminderThread...")
        threading.Thread(target=chatbot_logic.check_reminders, daemon=True, name="ReminderThreadStreamlit").start()
    if "VLCMonitorThreadStreamlit" not in thread_names:
        print("Streamlit: Starting VLCMonitorThread...")
        threading.Thread(target=chatbot_logic.vlc_state_monitor, daemon=True, name="VLCMonitorThreadStreamlit").start()
    
    print("Streamlit: Chatbot backend (DB and threads) initialized.")
    return True

try:
    chatbot_initialized = initialize_chatbot_backend_streamlit()
    if chatbot_logic.model is None: # Check if model loading failed
        st.error("Critical Error: Failed to load NLU model. The assistant cannot function. Please check the console logs.")
        st.stop()
except Exception as e_init:
    st.error(f"Critical Error during initialization: {e_init}. The assistant cannot function. Please check the console logs.")
    st.stop()


# --- CSS مخصص ---
st.markdown("""
<style>
    /* ... (نفس الـ CSS الرائع الذي قدمته سابقًا) ... */
    .stApp { background-color: #282c34; color: #abb2bf; } /* Dark theme base */
    .stChatInputContainer > div > input { 
        border-radius: 10px; 
        padding: 0.75rem 1rem; 
        background-color: #3a3f4b;
        color: #abb2bf;
        border: 1px solid #4f5666;
    }
    .stButton>button { 
        border-radius: 10px; 
        border: 1px solid #61afef; 
        background-color: #61afef;
        color: #282c34;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #528bce;
        border-color: #528bce;
    }
    [data-testid="stChatMessage"] {
        border-radius: 12px; 
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.75rem;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        border: 1px solid #4f5666;
    }
    [data-testid="stChatMessageContent"] p { margin-bottom: 0.3em; line-height: 1.6; }
    .stChatMessage[data-testid="stChatMessage"][role="user"] {
        background-color: #61afef; /* Brighter blue for user */
        color: #282c34; /* Darker text for contrast */
        margin-left: auto; 
    }
    .stChatMessage[data-testid="stChatMessage"][role="assistant"] {
        background-color: #3a3f4b; /* Darker assistant bubble */
        color: #abb2bf;
        margin-right: auto; 
    }
    .guide-section h2, .guide-section h3 { color: #61afef; margin-top:1.5em; border-bottom: 1px solid #4f5666; padding-bottom: 0.3em;}
    .stSidebar { padding-top: 1rem; background-color: #21252b; } 
    .stSidebar [data-testid="stMarkdownContainer"] p, .stSidebar [data-testid="stMarkdownContainer"] li, .stSidebar [data-testid="stText"], .stSidebar [data-testid="stSubheader"] {
        color: #abb2bf !important;
    }
    .stSidebar .stButton>button { width: 100%; margin-top: 0.5em; }
    .stExpander { border: 1px solid #4f5666; border-radius: 8px; }
    .stExpander header { font-weight: bold; }

    /* Style for the audio player */
    div.stAudio > audio {
        background-color: #4f5666; /* Darker background for the player */
        border-radius: 25px; /* Rounded player */
    }
    /* Attempt to style audio player controls (browser dependent) */
    div.stAudio > audio::-webkit-media-controls-panel {
        background-color: #4f5666;
        border-radius: 25px;
    }
    div.stAudio > audio::-webkit-media-controls-play-button,
    div.stAudio > audio::-webkit-media-controls-volume-slider-container,
    div.stAudio > audio::-webkit-media-controls-current-time-display,
    div.stAudio > audio::-webkit-media-controls-time-remaining-display {
        filter: invert(1) brightness(1.5); /* Make controls lighter on dark background */
    }


</style>
""", unsafe_allow_html=True)


# --- الشريط الجانبي (Sidebar) ---
with st.sidebar:
    # يمكنك وضع شعار هنا إذا أردت
    # try:
    #     logo_path = os.path.join(chatbot_logic.script_dir, "logo.png") 
    #     if os.path.exists(logo_path):
    #         image = Image.open(logo_path)
    #         st.image(image, width=120, use_column_width='auto') # Center logo
    # except Exception as e:
    #     print(f"Error loading logo: {e}")
    st.markdown("<h1 style='text-align: center; color: #61afef; font-size: 2.5em;'>🤖</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>AI Assistant Pro</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.button("➕ New Chat", key="new_chat_sidebar_button"):
        st.session_state.messages = [] 
        st.session_state.chat_id = f"chat_{int(time.time())}_{random.randint(1000,9999)}" 
        st.session_state.last_assistant_context_streamlit = None 
        hour = int(datetime.datetime.now().hour)
        if 0 <= hour < 12: welcome_message = "Good Morning! Starting a new chat. How can I help?"
        elif 12 <= hour < 18: welcome_message = "Good Afternoon! New chat. What can I do for you?"
        else: welcome_message = "Good Evening! Fresh chat. How may I assist?"
        
        new_chat_audio_path = None
        try:
            tts_new_chat = gTTS(text=welcome_message, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', dir=tempfile.gettempdir()) as fp_new:
                new_chat_audio_path = fp_new.name
            tts_new_chat.save(new_chat_audio_path)
        except Exception as e_new_chat_tts: print(f"Error generating new chat welcome TTS: {e_new_chat_tts}")
        st.session_state.messages.append({"role": "assistant", "content": welcome_message, "audio_path": new_chat_audio_path})
        st.rerun()


    st.markdown("---")
    st.subheader("About")
    st.info("Your intelligent assistant for music, reminders, web searches, and more. Built with Python & Streamlit.")
    st.markdown("---")
    st.subheader("User Guide")
    with st.expander("General", expanded=False): st.markdown("- `hello`\n- `what can you do?`\n- `tell me a fun fact`")
    with st.expander("Music"): st.markdown("- `play [song by artist]`\n- `stop music`, `pause`, `resume`, `next song`")
    with st.expander("Info"): st.markdown("- `weather in [city]`\n- `summarize [topic]`\n- `detailed search for [topic]`\n- `google [query]`\n- `time now`\n- `where am I?`")
    with st.expander("Tasks"): st.markdown("- `add reminder [task] [date] [time]`\n- `show reminders`\n- `open/close notepad`\n- `screenshot`")
    with st.expander("System"): st.markdown("- `system info`\n- `increase/decrease/mute/unmute volume`")
    with st.expander("Translate"): st.markdown("- `translate [text] to [language]`")
    st.markdown("---")
    st.caption("© 2024 Abdallah AI")


# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Initial welcome messages with audio
    welcome_msgs_with_audio = []
    hour = int(datetime.datetime.now().hour)
    welcome_text_1 = ("Good Morning" if 0 <= hour < 12 else "Good Afternoon" if 12 <= hour < 18 else "Good Evening") + " Abdallah! How can I help you today?"
    
    try:
        tts_wc1 = gTTS(text=welcome_text_1, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', dir=tempfile.gettempdir()) as fp: wc1_path = fp.name
        tts_wc1.save(wc1_path)
        welcome_msgs_with_audio.append({"role": "assistant", "content": welcome_text_1, "audio_path": wc1_path})
    except Exception as e: welcome_msgs_with_audio.append({"role": "assistant", "content": welcome_text_1, "audio_path": None}); print(f"TTS Error: {e}")

    welcome_text_2 = random.choice(["I'm ready for your commands.", "How can I assist you?", "Listening for your command."])
    try:
        tts_wc2 = gTTS(text=welcome_text_2, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', dir=tempfile.gettempdir()) as fp: wc2_path = fp.name
        tts_wc2.save(wc2_path)
        welcome_msgs_with_audio.append({"role": "assistant", "content": welcome_text_2, "audio_path": wc2_path})
    except Exception as e: welcome_msgs_with_audio.append({"role": "assistant", "content": welcome_text_2, "audio_path": None}); print(f"TTS Error: {e}")
    
    st.session_state.messages.extend(welcome_msgs_with_audio)


if "chat_id" not in st.session_state: 
    st.session_state.chat_id = f"chat_{int(time.time())}_{random.randint(1000,9999)}"


# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        for part in str(message["content"]).split('\n'): # Ensure content is string
            if part.strip(): st.markdown(part)
        
        if message.get("audio_path") and message["role"] == "assistant":
            # Only attempt to play if the file exists; helps with reruns if temp files are cleaned
            if os.path.exists(message["audio_path"]):
                try:
                    with open(message["audio_path"], "rb") as audio_file:
                        # Unique key for each audio player
                        st.audio(audio_file.read(), format="audio/mp3", key=f"audio_{st.session_state.chat_id}_{i}")
                except FileNotFoundError:
                    # This can happen if a rerun occurs after a temp file was deleted
                    # Or if the path was stored but the file creation failed initially
                    print(f"Audio file not found for message {i}: {message['audio_path']}")
                    # st.caption("(Audio file was removed or unavailable)") # Optional: inform user
                except Exception as e_audio_play:
                    print(f"Error playing audio file {message['audio_path']} in Streamlit: {e_audio_play}")
                    st.caption("(Audio playback error)")
            # else:
                # print(f"Audio path {message['audio_path']} does not exist for message {i}")


# Chat input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "audio_path": None})
    with st.chat_message("user"): # Display user message immediately
        st.markdown(prompt)

    # Assistant's turn
    with st.chat_message("assistant"):
        message_placeholder = st.empty() 
        message_placeholder.markdown("Thinking...") # Show thinking message
        
        current_context_for_logic = st.session_state.get('last_assistant_context_streamlit', None)
        
        # Call the backend logic
        assistant_reply_text, updated_context = chatbot_logic.get_chatbot_response(prompt, current_context_for_logic)
        
        # Update Streamlit's session state with the context returned from the backend
        st.session_state.last_assistant_context_streamlit = updated_context

        message_placeholder.markdown(assistant_reply_text) # Update placeholder with text reply

        temp_audio_file_path_for_current_reply = None
        audio_generated_this_turn = False
        if assistant_reply_text and assistant_reply_text.strip():
            try:
                text_for_tts = assistant_reply_text
                text_for_tts = re.sub(r"\(Suppressed due to music:.*?\)", "", text_for_tts).strip()
                
                if text_for_tts:
                    print(f"Streamlit: Generating TTS for: '{text_for_tts}'")
                    tts_web = gTTS(text=text_for_tts, lang='en', slow=False)
                    # Use a more robust way to get a temporary directory if default causes issues
                    temp_dir = tempfile.gettempdir()
                    temp_audio_file_path_for_current_reply = os.path.join(temp_dir, f"speech_{int(time.time()*1000)}.mp3")
                    tts_web.save(temp_audio_file_path_for_current_reply)
                    audio_generated_this_turn = True
            except Exception as e_gtts_st:
                print(f"Error generating TTS in Streamlit: {e_gtts_st}")
                # Update the displayed text if audio generation fails
                assistant_reply_text_with_error = assistant_reply_text + " (Audio generation failed)"
                message_placeholder.markdown(assistant_reply_text_with_error) 
        
        # Add the assistant's full response (text + audio path) to session_state messages *after* displaying text
        # This makes it part of the history for the next full re-render.
        assistant_message_data = {
            "role": "assistant", 
            "content": assistant_reply_text, # Original text, or text with error appended
            "audio_path": temp_audio_file_path_for_current_reply if audio_generated_this_turn else None
        }
        st.session_state.messages.append(assistant_message_data)

        # If audio was generated for *this current turn*, play it using st.audio
        # This st.audio will be part of the current message bubble.
        if audio_generated_this_turn and temp_audio_file_path_for_current_reply:
            try:
                with open(temp_audio_file_path_for_current_reply, "rb") as audio_file_data:
                    # This audio player is specifically for the *current* response
                    st.audio(audio_file_data.read(), format="audio/mp3", start_time=0)
            except Exception as e_audio_st:
                print(f"Error playing generated audio immediately in current bubble: {e_audio_st}")
                # Text is already displayed by message_placeholder

        # No st.rerun() here to allow audio to play without interruption.
        # The next user interaction or a more sophisticated update mechanism would refresh the full list.
        # However, since we append to st.session_state.messages, the new message *will* be in history.
        # If the audio player for the latest message doesn't show up immediately, a rerun might be needed,
        # but it often interrupts audio. This is a common Streamlit challenge.

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>AI Assistant Interface v1.3</p>", unsafe_allow_html=True)