'use client';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

export default function AudioGame() {
  const router = useRouter();
  const [timeLeft, setTimeLeft] = useState(10);

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          clearInterval(timer);
          router.push('/analysis'); // Go to analysis when done
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [router]);

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 bg-emerald-50">
      <h2 className="text-3xl font-bold text-gray-800 text-center mb-4">Word Blitz!</h2>
      <p className="text-xl text-center text-gray-600 mb-8">Name 3 types of drinks you order at the Kopitiam. Quick!</p>
      <div className={`text-7xl font-black mb-8 ${timeLeft <= 3 ? 'text-red-500 animate-bounce' : 'text-emerald-600'}`}>{timeLeft}s</div>
      <div className="mt-8 flex gap-2 items-center">
        <div className="w-4 h-4 bg-red-500 rounded-full animate-ping"></div>
        <div className="text-red-500 font-bold uppercase tracking-widest">Recording</div>
      </div>
    </div>
  );
}
