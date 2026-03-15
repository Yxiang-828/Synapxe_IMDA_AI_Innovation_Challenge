'use client';
import { useRouter } from 'next/navigation';

export default function Landing() {
  const router = useRouter();
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8 bg-emerald-50 text-center">
      <h1 className="text-4xl font-bold text-emerald-700 mb-4">MERaLiON</h1>
      <p className="text-gray-600 mb-12">Your empathetic, voice-driven health companion.</p>
      <button 
        onClick={() => router.push('/setup')}
        className="bg-emerald-600 text-white w-full py-4 rounded-full font-bold text-lg hover:bg-emerald-700 transition shadow-lg"
      >
        Get Started
      </button>
    </div>
  );
}