'use client';
import { useRouter } from 'next/navigation';

export default function Setup() {
  const router = useRouter();
  return (
    <div className="flex-1 p-8 flex flex-col bg-white">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Patient Profile</h2>
      <div className="space-y-4 flex-1">
        <div>
          <label className="block text-gray-600 mb-2 font-semibold">Nickname</label>
          <input type="text" defaultValue="Uncle Lim" className="w-full p-3 border rounded-xl" />
        </div>
        <div>
          <label className="block text-gray-600 mb-2 font-semibold">Chronic Condition to Monitor</label>
          <select className="w-full p-3 border rounded-xl bg-white">
            <option>Hypertension / Stroke Risk</option>
            <option>Dementia</option>
          </select>
        </div>
      </div>
      <button 
        onClick={() => router.push('/chat')}
        className="bg-emerald-600 text-white w-full py-4 rounded-full font-bold text-lg hover:bg-emerald-700 shadow-lg"
      >
        Start Monitoring
      </button>
    </div>
  );
}
