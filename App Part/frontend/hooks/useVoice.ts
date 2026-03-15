'use client';

import { useCallback, useRef } from 'react';

// Common mapping of text to audio filename
const AUDIO_MAP: Record<string, string> = {
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
};

export function useVoice() {
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const speak = useCallback((text: string, forceDynamic: boolean = false) => {
    // Stop any currently playing audio
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    window.speechSynthesis.cancel();

    // Check if we have a pre-recorded file for this exact text
    const audioFile = AUDIO_MAP[text];

    if (!forceDynamic && audioFile) {
      // Play pre-recorded MP3
      const audio = new Audio(`/audio/${audioFile}`);
      audioRef.current = audio;
      audio.play().catch(e => {
        console.warn("Audio play blocked by browser policy. Falling back to native TTS.", e);
        fallbackToNative(text);
      });
    } else {
      fallbackToNative(text);
    }
  }, []);

  const fallbackToNative = (text: string) => {
    // Clean up text if it contains variables like `[1]`
    const cleanText = text.replace(/\[\d+\]/, '');

    const utterance = new SpeechSynthesisUtterance(cleanText);
    const voices = window.speechSynthesis.getVoices();
    const preferredVoice = voices.find(v => v.name.includes('Female') || v.name.includes('Google UK'));
    if (preferredVoice) utterance.voice = preferredVoice;
    
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  };

  return { speak };
}
