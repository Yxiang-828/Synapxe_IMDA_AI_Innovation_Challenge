'use client';
import { useRouter } from 'next/navigation';

export default function CameraGame() {
  const router = useRouter();
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 bg-emerald-50">
      <h2 className="text-3xl font-bold text-gray-800 text-center mb-4">Camera Game</h2>
      <p className="text-xl text-center text-gray-600 mb-8">Coming soon...</p>
      <button 
        onClick={() => router.push('/chat')}
        className="bg-emerald-600 text-white w-full py-4 rounded-full font-bold text-lg hover:bg-emerald-700 shadow-lg"
      >
        Back to Chat
      </button>
    </div>
  );
}
