import os
import numpy as np
import soundfile as sf
import pyttsx3

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "audio_input")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_speech_sample(filename, text, rate=150, volume=1.0):
    """
    Generates a speech sample using TTS.
    rate: speed of speech (lower = slower/depressed/fatigued)
    volume: volume level (lower = quieter)
    """
    try:
        engine = pyttsx3.init()
        
        # Configure voice properties
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)
        
        # Save to file
        output_path = os.path.join(OUTPUT_DIR, filename)
        # pyttsx3 save_to_file is async-ish, need to runAndWait
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        
        if os.path.exists(output_path):
            print(f"Generated TTS: {output_path} (Rate: {rate}, Vol: {volume})")
        else:
            raise Exception("File not created by pyttsx3")

    except Exception as e:
        print(f"TTS failed for {filename}: {e}. Fallback to sine wave.")
        # Fallback: Generate simple sine wave
        fs = 16000
        duration = 3.0 # seconds
        f = rate * 2 # map rate to frequency somewhat
        t = np.linspace(0, duration, int(fs*duration))
        audio = 0.5 * np.sin(2 * np.pi * f * t)
        
        output_path = os.path.join(OUTPUT_DIR, filename)
        sf.write(output_path, audio, fs)
        print(f"Generated Sine Wave: {output_path}")

def main():
    print("Generating simulated patient samples...")
    
    # 1. Healthy Sample
    # Normal speed (200), Normal volume (1.0)
    generate_speech_sample(
        "sample_healthy.wav", 
        "The quick brown fox jumps over the lazy dog. I feel great today and have lots of energy.",
        rate=200, 
        volume=1.0
    )
    
    # 2. Depression Sample 
    # Slow speed (100), Low volume (0.5), Monotone content
    generate_speech_sample(
        "sample_depression.wav",
        "I feel very tired and sad. I don't want to do anything today. Everything is gray.",
        rate=100,
        volume=0.5
    )
    
    # 3. Cognitive Decline (MCI) Sample
    # Very slow (80), Confusion
    generate_speech_sample(
        "sample_mci.wav",
        "I... I think... the dog... went over... the... um... what was it? The cat?",
        rate=80,
        volume=0.8
    )

    # 4. Voice Fatigue Sample
    # Normal speed but maybe we can simulate something else (hard with TTS, using standard settings)
    generate_speech_sample(
        "sample_fatigue.wav",
        "My throat hurts and my voice is raspy. I have been talking all day.",
        rate=180, 
        volume=0.7
    )

    print(f"\nDone! Samples saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
