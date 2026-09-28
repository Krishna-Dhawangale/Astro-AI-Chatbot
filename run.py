"""
Astrology AI Chatbot - Server Launcher
Run this from the astro_env root with the VENV Python:

    Scripts\python.exe run.py
"""
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True, reload_dirs=[BASE_DIR])
