"""
colab_runner.py
---------------
Run GCC Derivatives Terminal in Google Colab via ngrok tunnel.

Usage — paste this cell into Colab and run:

    !git clone https://github.com/YOUR_USERNAME/gcc-derivatives.git
    %cd gcc-derivatives
    !pip install -q -r requirements.txt pyngrok
    !python colab_runner.py

Then click the ngrok URL that appears in the output.
"""

import subprocess
import sys
import time

# ── Install pyngrok if missing ────────────────────────────────────────────────
try:
    from pyngrok import ngrok
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyngrok"])
    from pyngrok import ngrok

# ── Optional: set your ngrok authtoken for longer sessions ───────────────────
# ngrok.set_auth_token("YOUR_NGROK_TOKEN")

# ── Launch Streamlit in background ───────────────────────────────────────────
print("Starting Streamlit server…")
proc = subprocess.Popen(
    ["streamlit", "run", "app.py",
     "--server.port", "8501",
     "--server.headless", "true",
     "--server.enableCORS", "false",
     "--server.enableXsrfProtection", "false",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
time.sleep(4)

# ── Open ngrok tunnel ─────────────────────────────────────────────────────────
public_url = ngrok.connect(8501)
print(f"\n✅  GCC Derivatives Terminal is live at:\n\n    {public_url}\n")
print("Keep this cell running. Interrupt to stop.")

try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    ngrok.disconnect(public_url)
    print("Server stopped.")
