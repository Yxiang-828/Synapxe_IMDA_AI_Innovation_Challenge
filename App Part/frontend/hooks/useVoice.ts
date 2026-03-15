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
    // Try to pick the most feminine voice available
    let preferredVoice = voices.find(v => v.name.toLowerCase().includes('female'));
    if (!preferredVoice) {
      // Try Google voices with 'en' and 'female' or 'english' in the name
      preferredVoice = voices.find(v => (v.name.toLowerCase().includes('google') && (v.name.toLowerCase().includes('female') || v.name.toLowerCase().includes('english'))));
    }
    if (!preferredVoice) {
      // Fallback: pick the highest-pitch voice
      preferredVoice = voices.sort((a, b) => (b.pitch || 1) - (a.pitch || 1))[0];
    }
    if (preferredVoice) utterance.voice = preferredVoice;
    utterance.rate = 1.0;
    utterance.pitch = 1.4; // Higher pitch for more feminine sound
    window.speechSynthesis.speak(utterance);
  };

  return { speak };
}
