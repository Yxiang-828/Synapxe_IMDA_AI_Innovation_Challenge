import './globals.css';
import Script from 'next/script';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <Script src="https://telegram.org/js/telegram-web-app.js" strategy="beforeInteractive" />
      </head>
      <body className="flex items-center justify-center min-h-screen bg-gradient-to-br from-indigo-950 via-purple-900 to-slate-900">
        {/* The Universal Mobile Phone Container */}
        <div className="w-full max-w-sm h-[820px] bg-gray-50 rounded-[3rem] shadow-[0_0_50px_-12px_rgba(0,0,0,0.8)] overflow-hidden border-[14px] border-black flex flex-col relative ring-4 ring-gray-800/50">
          
          {/* Hardware "Notch" UI (Dynamic Island style) */}
          <div className="absolute top-0 inset-x-0 h-7 flex justify-center z-50 pointer-events-none">
            <div className="w-32 h-6 bg-black rounded-b-3xl shadow-inner"></div>
          </div>

          {/* Modern App Header */}
          <div className="bg-gradient-to-r from-emerald-500 to-teal-600 p-6 pt-9 shadow-md flex items-center justify-center gap-3 z-40 relative">
            <span className="text-white font-extrabold text-2xl tracking-tight leading-none drop-shadow-sm">
              MERaLiON 
            </span>
            <span className="text-3xl animate-bounce drop-shadow-md origin-bottom inline-block">😁</span>
          </div>

          {/* Page Content */}
          <div className="flex-1 relative overflow-hidden flex flex-col">
            {children}
          </div>
        </div>
      </body>
    </html>
  );
}