'use client';
import { useState, useRef } from 'react';
import { useRouter } from 'next/navigation';

export default function Chat() {
  const router = useRouter();
  const [messages, setMessages] = useState([{ sender: 'llm', text: 'Eh, time to catch up! How are you feeling today?' }]);
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<any>(null);

  const startListening = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) return alert("Browser doesn't support speech recognition.");
    
    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.continuous = false;
    recognitionRef.current.interimResults = false;

    recognitionRef.current.onstart = () => setIsListening(true);
    
    recognitionRef.current.onresult = async (event: any) => {
      const text = event.results[0][0].transcript;
      setIsListening(false);
      
      setMessages(prev => [...prev, { sender: 'user', text }]);
      
      // Fetch from backend
      const res = await fetch('http://localhost:8080/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      
      setMessages(prev => [...prev, { sender: 'llm', text: data.reply }]);
      
      const utterance = new SpeechSynthesisUtterance(data.reply);
      window.speechSynthesis.speak(utterance);

      // Route to games based on backend response
      if (data.route) {
        setTimeout(() => router.push(data.route), 4000); 
      }
    };

    recognitionRef.current.start();
  };

  return (
    <div className="flex-1 flex flex-col bg-gray-100">
      <div className="flex-1 p-4 overflow-y-auto space-y-4">
        {messages.map((msg, i) => (
          <div key={i} className={`p-4 rounded-2xl max-w-[85%] ${msg.sender === 'user' ? 'bg-emerald-100 self-end ml-auto' : 'bg-white shadow-md'}`}>
            <p className="text-gray-800 text-lg">{msg.text}</p>
          </div>
        ))}
      </div>
      <div className="p-4 bg-white border-t space-y-2">
        <div className="flex gap-2 justify-center">
          <button 
            onClick={() => router.push('/audio-game')}
            className="bg-blue-500 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-600"
          >
            Audio Game
          </button>
          <button 
            onClick={() => router.push('/camera-game')}
            className="bg-purple-500 text-white px-4 py-2 rounded-lg text-sm hover:bg-purple-600"
          >
            Camera Game
          </button>
          <button 
            onClick={() => router.push('/analysis')}
            className="bg-green-500 text-white px-4 py-2 rounded-lg text-sm hover:bg-green-600"
          >
            Analysis
          </button>
        </div>
      </div>
      <div className="p-6 bg-white border-t flex justify-center pb-8">
        <button 
          onClick={startListening}
          className={`w-20 h-20 rounded-full flex items-center justify-center shadow-lg transition-all ${isListening ? 'bg-red-500 animate-pulse' : 'bg-emerald-600 hover:bg-emerald-700'}`}
        >
          <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path></svg>
        </button>
      </div>
    </div>
  );
}
