import os
from gtts import gTTS

AUDIO_DIR = "App Part/frontend/public/audio"

if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# Mapping from text to audio filename as defined in our useVoice hook
AUDIO_MAP = {
  "Smile as wide as you can!": "smile_wide.mp3",
  "Keep smiling, don't relax!": "keep_smiling.mp3",
  "Hold still, keep calm": "hold_still.mp3",
  "Get Ready!": "get_ready.mp3",
  "Good resting pos! Now stand up fully.": "stand_up_fully.mp3",
  "Great! Now sit back down.": "sit_back_down.mp3",
  "Good rep! Stand up again.": "stand_up_again.mp3",
  "Keep raising! Reach for the sky.": "reach_sky.mp3",
  "Excellent height! Lower arm slowly.": "lower_arm.mp3",
  "Great form! Raise again.": "raise_again.mp3",
  "Measurement complete": "measurement_complete.mp3"
}

def generate_audio():
    print("Generating voice lines using Google TTS (gTTS)...")
    print("For Piper TTS: replace these files with Piper-generated ones later.")
    
    for text, filename in AUDIO_MAP.items():
        filepath = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Generating: {filename} -> '{text}'")
            tts = gTTS(text=text, lang='en', tld='co.uk', slow=False) # co.uk provides a nice soothing female-ish voice by default
            tts.save(filepath)
        else:
            print(f"Skipping: {filename} (already exists)")
            
    print("Done! All audio files generated.")

if __name__ == "__main__":
    generate_audio()
