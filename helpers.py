import secrets
import os
import re
import time 
import logging
from datetime import datetime, timedelta


from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from inf import FILE_DELETION_TIME


logger = logging.getLogger(__name__)


def escape_markdown(text):
    escape_chars = '_*[]()~`>#+-=|{}.!'
    return ''.join('\\' + char if char in escape_chars else char for char in text)

def secure_filename(filename):
    """Sanitizes and secures a filename."""
    filename = re.sub(r'[\\/]', '_', filename)  # Sanitize
    return f"{secrets.token_hex(16)}_{''.join(c for c in filename if c.isalnum() or c in '_.')}"

def validate_user_input(text):
    """Validates and sanitizes user input."""
    import bleach
    sanitized_text = bleach.clean(text, tags=[], strip=True)
    return sanitized_text

def is_safe_path(basedir, path, follow_symlinks=True):
    if follow_symlinks:
        return os.path.realpath(path).startswith(os.path.realpath(basedir))
    return os.path.abspath(path).startswith(os.path.abspath(basedir))

def generate_callback_data(action, session_id):
    """Generates callback data."""
    return f"{action}:{session_id}"

def parse_callback_data(callback_data):
    try:
        action, session_id = callback_data.split(':')
        return action, session_id
    except ValueError:
        raise ValueError("Invalid callback data format")
    
def cleanup_old_files():
    current_time = time.time()
    for root, dirs, files in os.walk('uploads'):
        for file in files:
            file_path = os.path.join(root, file)
            if current_time - os.path.getmtime(file_path) > FILE_DELETION_TIME:
                os.remove(file_path)

def generate_main_keyboard(session_id: str) -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("Remove", callback_data=generate_callback_data("remove", session_id))],
        [InlineKeyboardButton("Apply Contrast", callback_data=generate_callback_data("contrast", session_id))],
        [InlineKeyboardButton("Black & White", callback_data=generate_callback_data("bw", session_id))],
        [InlineKeyboardButton("Upscale 33%", callback_data=generate_callback_data("upscale", session_id))],
        [InlineKeyboardButton("Downscale 33%", callback_data=generate_callback_data("downscale", session_id))],
        [InlineKeyboardButton("Darker Enhance", callback_data=generate_callback_data("darker", session_id))],
        [InlineKeyboardButton("Local Contrast", callback_data=generate_callback_data("local contrast", session_id))]
    ]
    return InlineKeyboardMarkup(keyboard)
