import subprocess
import re
import time
import sys
import urllib.request

NEXT_JS_URL = "http://127.0.0.1:3000"

print('Aiko 💕: Pinging Next.js to make sure it is actually awake before we tunnel...')

up = False
for _ in range(60):
    try:
        resp = urllib.request.urlopen(NEXT_JS_URL)
        if resp.status == 200:
            up = True
            print('Aiko 💕: Next.js is responding! Booting up the Cloudflare tunnel...')
            break
    except Exception:
        pass
    print("Waiting for Next.js to finish compiling...")
    time.sleep(2)

if not up:
    print('Aiko 💕: Error! Next.js never started properly on port 3000. Aborting tunnel binding.')
    sys.exit(1)

# Start cloudflared via HTTP2 to avoid QUIC/UDP issues
tunnel_proc = subprocess.Popen(
    [
        "cloudflared",
        "tunnel",
        "--protocol",
        "http2",
        "--url",
        NEXT_JS_URL,
        "--http-host-header",
        "localhost",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

new_url = None
for line in tunnel_proc.stdout:
    print("[TUNNEL]", line.strip(), flush=True)
    if not new_url:
        match = re.search(r"(https://[a-zA-Z0-9-]+\.trycloudflare\.com)", line)
        if match:
            new_url = match.group(1)
            break

if not new_url:
    print("Failed to get tunnel URL!")
    sys.exit(1)

print(f"Found Tunnel URL: {new_url}")

bot_file = "bot.py"
with open(bot_file, "r", encoding="utf-8") as f:
    content = f.read()

content = re.sub(r'BASE_URL\s*=\s*(["\']).*?\1', f'BASE_URL = "{new_url}"', content)
with open(bot_file, "w", encoding="utf-8") as f:
    f.write(content)

print("Updated bot.py with new URL!")
print("Starting Telegram Bot...")
try:
    bot_proc = subprocess.Popen([sys.executable, "bot.py"])
    bot_proc.wait()
except KeyboardInterrupt:
    print("Shutting down gracefully...")
finally:
    try:
        bot_proc.terminate()
    except Exception:
        pass
    try:
        tunnel_proc.terminate()
    except Exception:
        pass