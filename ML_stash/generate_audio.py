import numpy as np
import scipy.io.wavfile as wav

# Generate a 5-second sine wave at 440 Hz
duration = 5.0
sample_rate = 16000
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)

# Add some noise to make it realistic
noise = np.random.normal(0, 0.1, audio_data.shape)
audio_data = audio_data + noise

# Normalize to 16-bit PCM range
audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

wav.write("sample.wav", sample_rate, audio_data)
print("sample.wav created successfully.")
