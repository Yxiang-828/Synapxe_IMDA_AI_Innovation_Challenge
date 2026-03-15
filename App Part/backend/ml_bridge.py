import os
import torch
import soundfile as sf
import subprocess
from transformers import pipeline

# We keep this at module level so it only loads once!
_transcriber = None

def get_transcriber():
    global _transcriber
    if _transcriber is None:
        print("Loading Whisper ASR Model... (This will take a few seconds on first run)")
        _transcriber = pipeline(
            "automatic-speech-recognition", 
            model="openai/whisper-tiny.en", 
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    return _transcriber

def get_ffmpeg_path() -> str:
    """Returns the path to the local project's ffmpeg.exe, or default 'ffmpeg' if in system PATH."""
    import shutil
    import glob
    
    # 1. Check system path (if terminal was restarted)
    if shutil.which("ffmpeg"):
        return "ffmpeg"
        
    # 2. Check Local Path Fallback (if they put it manually)
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ffmpeg", "bin", "ffmpeg.exe")
    if os.path.exists(local_path):
        return local_path
        
    # 3. Check Winget Packages (dynamically find version without needing regex)
    winget_dir = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.exists(winget_dir):
        # find the gyan ffmpeg folder explicitly
        ffmpeg_folders = glob.glob(os.path.join(winget_dir, "Gyan.FFmpeg*"))
        if ffmpeg_folders:
            # check inside for bin\ffmpeg.exe
            for f_folder in ffmpeg_folders:
                built_bins = glob.glob(os.path.join(f_folder, "ffmpeg*full_build", "bin", "ffmpeg.exe"))
                if built_bins:
                    return built_bins[0]
                    
    # Fallback to pure 'ffmpeg' so it catches basic errors
    return "ffmpeg"

# INJECT FFMPEG INTO PATH IMMEDIATELY SO TRANSFORMERS CAN FIND IT
_found_ffmpeg = get_ffmpeg_path()
if _found_ffmpeg != "ffmpeg" and os.path.exists(_found_ffmpeg):
    ffmpeg_dir = os.path.dirname(_found_ffmpeg)
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] += os.pathsep + ffmpeg_dir

def convert_oga_to_wav(input_path: str, output_path: str) -> bool:
    """
    Converts Telegram's .oga/.ogg voice notes into .wav using a direct system call
    to FFmpeg.
    """
    ffmpeg_exe = get_ffmpeg_path()
    try:
        command = [ffmpeg_exe, '-y', '-i', input_path, '-ar', '16000', '-ac', '1', output_path]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except FileNotFoundError:
        print(f"FFMPEG NOT FOUND at {ffmpeg_exe}. Please run .\\build.bat to install it via winget.")
        return False
    except Exception as e:
        print(f"FFMPEG Error: {e}")
        return False


def transcribe_audio(file_path: str) -> str:
    """
    Takes a path to an audio file (.ogg from Telegram) and transcibes it to text.
    """
    try:
        # 1. Convert .oga to .wav first!
        wav_path = file_path.replace(".ogg", ".wav")
        success = convert_oga_to_wav(file_path, wav_path)
        
        if not success:
            return "[Could not transcribe audio - FFMPEG Missing]"
            
        # 2. Feed the clean .wav into HuggingFace pipeline
        transcriber = get_transcriber()
        result = transcriber(wav_path)
        
        # 3. Cleanup converted file
        if os.path.exists(wav_path):
            os.remove(wav_path)
            
        return result.get("text", "").strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return "[Could not transcribe audio]"

def analyze_voice_fatigue(file_path: str) -> float:
    """
    Mock function representing your ML_Stash logic.
    Returns a float out of 1.0 (Higher = more fatigue).
    For demo, we just randomly return a high fatigue score to trigger state 1.
    """
    # In the real version, we call librosa inside your VoiceFatigue scripts
    return 0.85
