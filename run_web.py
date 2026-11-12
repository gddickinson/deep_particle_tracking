"""Start the Deep Particle Tracker web app on http://127.0.0.1:8201 (local only).

    pip install -r requirements-web.txt
    python run_web.py
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("web.app:app", host="127.0.0.1", port=8201)
