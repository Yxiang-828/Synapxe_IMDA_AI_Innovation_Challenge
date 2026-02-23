'use client';

import { useState, useEffect, useRef } from 'react';

export default function MobileDemo() {
  const [messages, setMessages] = useState([
    { sender: 'llm', text: 'Eh, time to catch up! How are you feeling today?' }
  ]);
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [showCamera, setShowCamera] = useState(false);
  
  const recognitionRef = useRef<any>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Initialize Voice Recognition
  useEffect(() => {
  const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;    
  if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = true;

      recognitionRef.current.onresult = (event: any) => {
        const current = event.resultIndex;
        const currentTranscript = event.results[current][0].transcript;
        setTranscript(currentTranscript);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
        if (transcript) handleSend(transcript); 
      };
    }
  }, [transcript]);

  // Handle Camera Mounting for Game Category B
  const startCamera = async () => {
    setShowCamera(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
    }
    setShowCamera(false);
    setMessages((prev) => [...prev, { sender: 'llm', text: 'Looking good! Facial symmetry is normal. Rest well ok?' }]);
  };

  const toggleListen = () => {
    if (isListening) {
      recognitionRef.current?.stop();
    } else {
      setTranscript('');
      recognitionRef.current?.start();
      setIsListening(true);
    }
  };

  const handleSend = async (text: string) => {
    setMessages((prev) => [...prev, { sender: 'user', text }]);
    setTranscript('');

    try {
      const res = await fetch('http://localhost:8080/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      
      setMessages((prev) => [...prev, { sender: 'llm', text: data.reply }]);
      
      // Speak the response
      const utterance = new SpeechSynthesisUtterance(data.reply);
      utterance.rate = 1.05; 
      window.speechSynthesis.speak(utterance);

      // Trigger the camera game if the backend requests it
      if (data.triggerGame === 'emotion_mimic') {
        setTimeout(startCamera, 3000); // Wait for the LLM to finish speaking before opening camera
      }
      
    } catch (error) {
      console.error("Backend offline", error);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-900">
      <div className="w-full max-w-sm h-[800px] bg-gray-50 rounded-[3rem] shadow-2xl overflow-hidden border-8 border-gray-800 flex flex-col relative">
        
        <div className="bg-emerald-600 p-6 text-center text-white font-bold text-xl">
          MERaLiON Health
        </div>

        <div className="flex-1 p-4 overflow-y-auto space-y-4">
          {messages.map((msg, i) => (
            <div key={i} className={`p-4 rounded-2xl max-w-[85%] ${msg.sender === 'user' ? 'bg-emerald-100 self-end ml-auto' : 'bg-white shadow-md'}`}>
              <p className="text-gray-800 text-lg">{msg.text}</p>
            </div>
          ))}
          
          {transcript && (
            <div className="p-4 rounded-2xl max-w-[80%] bg-emerald-50 ml-auto italic text-gray-500">
              {transcript}...
            </div>
          )}

          {/* Embedded Camera UI for the Mini-Game */}
          {showCamera && (
            <div className="p-4 bg-white rounded-2xl shadow-md flex flex-col items-center animate-fade-in">
              <video ref={videoRef} autoPlay playsInline className="w-full rounded-xl bg-black mb-4" />
              <button onClick={stopCamera} className="bg-emerald-600 text-white px-4 py-2 rounded-full font-bold w-full hover:bg-emerald-700 transition">
                Capture & Analyze
              </button>
            </div>
          )}
        </div>

        <div className="p-6 bg-white border-t flex justify-center pb-8">
          <button 
            onClick={toggleListen}
            disabled={showCamera}
            className={`w-20 h-20 rounded-full flex items-center justify-center shadow-lg transition-all ${isListening ? 'bg-red-500 animate-pulse' : 'bg-emerald-600 hover:bg-emerald-700'} ${showCamera ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path></svg>
          </button>
        </div>

      </div>
    </div>
  );
}