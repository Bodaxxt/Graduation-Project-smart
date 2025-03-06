import os
import eel
@eel.expose
def playAssistantSound():
    try:
        if not os.path.exists(music_dir):
            print(f"Error: Sound file not found at {music_dir}")
            return

        pygame.mixer.init()  # Initialize the mixer module
        music_dir = r"E:\chatbooot\www\assets\audio\www_assets_audio_start_sound.mp3"
        print(f"Playing sound from: {music_dir}")
        pygame.mixer.music.load(music_dir)  # Load the audio file
        pygame.mixer.music.play()  # Play the audio
        while pygame.mixer.music.get_busy():  # Wait until playback is complete
            continue
        print("Assistant sound played successfully.")
    except Exception as e:
        print(f"Error playing sound: {e}")