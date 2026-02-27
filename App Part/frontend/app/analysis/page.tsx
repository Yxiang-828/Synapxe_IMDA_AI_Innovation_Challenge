'use client';
import { useRouter } from 'next/navigation';

export default function Analysis() {
  const router = useRouter();
  return (
    <div className="flex-1 p-6 flex flex-col bg-white">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 border-b pb-2">Health Dashboard</h2>
      
      <div className="bg-emerald-50 p-4 rounded-2xl mb-4 border border-emerald-200">
        <h3 className="font-bold text-emerald-800 mb-1">Latest Assessment</h3>
        <p className="text-emerald-600 text-sm">Completed 1 minute ago</p>
      </div>

      <div className="space-y-4 flex-1">
        <div className="p-4 bg-gray-50 rounded-xl flex justify-between items-center shadow-sm border">
          <span className="font-semibold text-gray-700">Speech Clarity</span>
          <span className="text-emerald-600 font-bold">98% (Normal)</span>
        </div>
        <div className="p-4 bg-gray-50 rounded-xl flex justify-between items-center shadow-sm border">
          <span className="font-semibold text-gray-700">Cognitive Recall</span>
          <span className="text-emerald-600 font-bold">Pass</span>
        </div>
      </div>

      <div className="bg-blue-50 text-blue-800 p-4 rounded-xl mb-6 text-sm text-center">
        No deterioration detected. Caretaker update skipped.
      </div>

      <button onClick={() => router.push('/camera-game')} className="bg-purple-600 text-white w-full py-4 rounded-full font-bold mb-2 hover:bg-purple-700 transition">
        Try Camera Game
      </button>

      <button onClick={() => router.push('/chat')} className="bg-gray-200 text-gray-800 w-full py-4 rounded-full font-bold hover:bg-gray-300 transition">
        Return to Home
      </button>
    </div>
  );
}
