# Vercel entrypoint - imports the Flask app from front.py
from front import app

# Export the app for Vercel
__all__ = ['app']
