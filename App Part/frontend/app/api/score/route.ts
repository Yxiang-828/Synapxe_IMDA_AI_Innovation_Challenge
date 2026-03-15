import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const body = await req.json();
    
    // Proxy the request from Next.js backend to the FastAPI backend running securely on localhost
    const res = await fetch('http://127.0.0.1:8080/api/log_score', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error proxying score:', error);
    return NextResponse.json({ error: 'Failed to proxy score' }, { status: 500 });
  }
}
