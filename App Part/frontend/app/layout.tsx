import './globals.css';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="flex items-center justify-center min-h-screen bg-gray-900">
        {/* The Universal Mobile Phone Container */}
        <div className="w-full max-w-sm h-[800px] bg-gray-50 rounded-[3rem] shadow-2xl overflow-hidden border-8 border-gray-800 flex flex-col relative">
          <div className="bg-emerald-600 p-6 text-center text-white font-bold text-xl">
            MERaLiON Health
          </div>
          {children}
        </div>
      </body>
    </html>
  );
}