# BOT.PY - BASE IMPORTS
import os
import sys
import time
import threading
import schedule
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.stats import skew
import aiohttp
import sqlite3

# TELEGRAM & CV2
import cv2
from PIL import Image
import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import matplotlib.pyplot as plt
from io import BytesIO

# SECURITY
import logging
from logging.handlers import RotatingFileHandler
import traceback
from werkzeug.utils import safe_join

# INTERNAL IMPORTS
from models import MODELS
from inf import current_model, middle_value, low_value_threshold, high_value_threshold, TOKEN, DEVELOPER_PASSWORD, MAX_REQUESTS, TIME_WINDOW
from helpers import escape_markdown, secure_filename, generate_callback_data, parse_callback_data, cleanup_old_files, generate_main_keyboard

# THREADING
executor = ThreadPoolExecutor(max_workers=4)  
# DEBUGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("bot.log", maxBytes=1024 * 1024 * 10, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

## DB FUNCTIONS
def init_db():
    with sqlite3.connect('rate_limit.db') as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS rate_limit 
            (user_id INTEGER, timestamp TEXT)
        ''')

def is_rate_limited(user_id):
    with sqlite3.connect('rate_limit.db') as conn:
        now = datetime.now()
        time_window = now - TIME_WINDOW
        conn.execute("DELETE FROM rate_limit WHERE timestamp < ?", (time_window.isoformat(),))
        count = conn.execute(
            "SELECT COUNT(*) FROM rate_limit WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        if count >= MAX_REQUESTS:
            return True
        conn.execute(
            "INSERT INTO rate_limit (user_id, timestamp) VALUES (?, ?)", (user_id, now.isoformat())
        )
    return False

### DOWNLOAD HELPER
async def download_file_with_retry(file, retries=3):
    for attempt in range(retries):
        try:
            return await file.download_as_bytearray()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e
## WELCOME FUNCTION
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = escape_markdown(update.effective_user.first_name)

    welcome_message = (
        f"Welcome *{user_name}* to SnapWave\n\n"
        "Please send your image as a file to avoid compression: *Press \\📎 then File\\>Gallery* \n\n"
        "⚠️ Do not send using only 'gallery' only, or your image will be compressed\\. ⚠️\n\n"
        "/start \\- Display this welcome message\n"
        "/remove \\- Remove your files from the system directly\\.\n\n"
        "IMPORTANT,  *Press \\📎 then File\\>Gallery*\\. You will get a corrected version of the image back\\.\n\n"
    )
    await update.message.reply_text(welcome_message, parse_mode='MarkdownV2')
    logger.info(f"User {update.effective_user.id} started the bot")


### ANALYSIS FUNCTION PHOTO
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Please send your photo as a file to avoid compression: Press 📎 then File>Gallery")
    logger.info(f"User {update.effective_user.id} attempted to send a photo")
    
    try:
        photo = update.message.photo[-1]
        photo_file = await photo.get_file()
        photo_data = await photo_file.download_as_bytearray()

        image = Image.open(BytesIO(photo_data)).convert('RGB')
        image_array = np.array(image)
    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        await update.message.reply_text("Sorry, I couldn't process the image. Please try again.")
        return

    # Initialize variables
    stats = {}
    colors = ('r', 'g', 'b')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    clip_values = {color: {'low': 255, 'high': 0} for color in colors}
    individual_histograms = []

    # Process each color channel
    for i, (color, ax_hist, ax_cdf) in enumerate(zip(colors, axes[0], axes[1])):
        channel_data = image_array[:, :, i].ravel()
        
        # Split data into dark and light halves
        dark_data = channel_data[channel_data < middle_value]
        light_data = channel_data[channel_data >= middle_value]
        
        hist, bin_edges = np.histogram(channel_data, bins=256, range=(0, 256))
        ax_hist.bar(bin_edges[:-1], hist, color=color, alpha=0.6, width=1)
        
        # Highlight the middle split
        ax_hist.axvline(middle_value, color='black', linestyle='--', alpha=0.5)
        
        max_hist_value = np.max(hist)
        for bin_index in range(256):
            if hist[bin_index] < low_value_threshold * max_hist_value:
                ax_hist.axvspan(bin_index, bin_index + 1, color='yellow', alpha=0.3)
                clip_values[color]['low'] = min(clip_values[color]['low'], bin_index)
            elif hist[bin_index] > high_value_threshold * max_hist_value:
                ax_hist.axvspan(bin_index, bin_index + 1, color='red', alpha=0.3)
                clip_values[color]['high'] = max(clip_values[color]['high'], bin_index)
        
        ax_hist.set_title(f'{color.upper()} Histogram')
        ax_hist.set_xlabel('Pixel Intensity')
        ax_hist.set_ylabel('Frequency')

        # Calculate statistics for dark and light halves
        stats[color] = {
            'dark': {
                'mean': np.mean(dark_data),
                'median': np.median(dark_data),
                'std_dev': np.std(dark_data),
                'skewness': skew(dark_data),
                'q1': np.percentile(dark_data, 20),
                'q3': np.percentile(dark_data, 80)
            },
            'light': {
                'mean': np.mean(light_data),
                'median': np.median(light_data),
                'std_dev': np.std(light_data),
                'skewness': skew(light_data),
                'q1': np.percentile(light_data, 20),
                'q3': np.percentile(light_data, 80)
            }
        }

        cdf = np.cumsum(hist) / np.sum(hist)
        ax_cdf.plot(bin_edges[:-1], cdf, color=color)
        ax_cdf.set_title(f'{color.upper()} CDF')
        ax_cdf.set_xlabel('Pixel Intensity')
        ax_cdf.set_ylabel('Cumulative Frequency')

        ax_cdf.axvline(middle_value, color='black', linestyle='--', alpha=0.5)
        ax_cdf.text(middle_value, 0.5, 'Split', rotation=90, verticalalignment='bottom')

        # Create individual histogram
        fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
        ax_ind.bar(bin_edges[:-1], hist, color=color, alpha=0.6, width=1)
        ax_ind.axvline(middle_value, color='black', linestyle='--', alpha=0.5)
        ax_ind.set_title(f'{color.upper()} Histogram (Split at 128)')
        ax_ind.set_xlabel('Pixel Intensity')
        ax_ind.set_ylabel('Frequency')
        for bin_index in range(256):
            if hist[bin_index] < low_value_threshold * max_hist_value:
                ax_ind.axvspan(bin_index, bin_index + 1, color='yellow', alpha=0.3)
            elif hist[bin_index] > high_value_threshold * max_hist_value:
                ax_ind.axvspan(bin_index, bin_index + 1, color='red', alpha=0.3)
        plt.tight_layout()
        ind_hist_io = BytesIO()
        fig_ind.savefig(ind_hist_io, format='png')
        ind_hist_io.seek(0)
        plt.close(fig_ind)
        individual_histograms.append((color, ind_hist_io))
    
    plt.tight_layout()
    plot_io = BytesIO()
    fig.savefig(plot_io, format='png')
    plot_io.seek(0)
    plt.close(fig)

    # Create analysis text
    analysis = []
    final_message = []
    overall_brightness = np.mean(image_array)
    overall_contrast = np.std(image_array)

    # Determine channel with lowest mean intensity
    lowest_intensity_channel = min(stats, key=lambda c: (stats[c]['dark']['mean'] + stats[c]['light']['mean']) / 2)
    
    for color, stat in stats.items():
        analysis.append(
            f"{color.upper()} Channel:\n"
            f"Dark Half (0-127):\n"
            f"  Mean: {stat['dark']['mean']:.2f}, Median: {stat['dark']['median']:.2f}\n"
            f"  StdDev: {stat['dark']['std_dev']:.2f}, Skew: {stat['dark']['skewness']:.2f}\n"
            f"  Q1: {stat['dark']['q1']:.2f}, Q3: {stat['dark']['q3']:.2f}\n"
            f"Light Half (128-255):\n"
            f"  Mean: {stat['light']['mean']:.2f}, Median: {stat['light']['median']:.2f}\n"
            f"  StdDev: {stat['light']['std_dev']:.2f}, Skew: {stat['light']['skewness']:.2f}\n"
            f"  Q1: {stat['light']['q1']:.2f}, Q3: {stat['light']['q3']:.2f}\n"
            f"Clip Range: {clip_values[color]['low']}-{clip_values[color]['high']}\n"
        )

    overall_analysis = (
        f"Overall Image Statistics:\n"
        f"Brightness: {overall_brightness:.2f}, Contrast: {overall_contrast:.2f}\n"
        f"Yellow highlight: <{low_value_threshold*100}% of max frequency\n"
        f"Red highlight: >{high_value_threshold*100}% of max frequency\n"
    )
    analysis_text = "\n".join(analysis) + overall_analysis
    
    # Determine the lowest intensity channel for personalized feedback
    feedback_message = []
    feedback_message.append(f"The {lowest_intensity_channel.upper()} channel has the lowest overall intensity, indicating that this color might dominate darker areas in the image or is underrepresented.")

    # Add specific feedback based on analysis
    for color, stat in stats.items():
        if stat['dark']['mean'] < 50 and stat['light']['mean'] < 180:
            feedback_message.append(f"The {color.upper()} channel has generally low intensity values, which might indicate a darker image or underexposure for this color.")
        elif stat['dark']['std_dev'] < 15 and stat['light']['std_dev'] < 15:
            feedback_message.append(f"The {color.upper()} channel exhibits low variance, suggesting the image may lack contrast for this color.")
        if clip_values[color]['low'] > 10 and clip_values[color]['high'] < 245:
            feedback_message.append(f"The {color.upper()} channel is missing significant spectrum ranges, possibly indicating poor color diversity or underexposure.")
    
    if not feedback_message:
        feedback_message.append("The image appears balanced in terms of color, brightness, and contrast.")
    
    overall_analysis_text = " ".join(feedback_message)

    # Send combined histogram plot
    await update.message.reply_photo(
        photo=plot_io, 
        caption="RGB histograms and CDFs with low/high value zone highlighting and split at 128."
    )
    
    # Send analysis as text
    await update.message.reply_text(f"Analysis:\n{analysis_text}")

    # Send individual color histograms
    for color, hist_io in individual_histograms:
        await update.message.reply_photo(
            photo=hist_io,
            caption=f"{color.upper()} channel histogram (split at 128)"
        )
        await asyncio.sleep(1)  # 1 second delay between histograms
    
    await update.message.reply_text(f"Final Analysis: {overall_analysis_text}")

### ANALYSIS OVER. PROCESSING STARTS HERE

## HANDLER FUNCTION
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if is_rate_limited(user_id):
        await update.message.reply_text("You're sending requests too quickly. Please wait a moment before trying again.")
        return

    document = update.message.document

    if document.file_size > 20 * 1024 * 1024:
        await update.message.reply_text("File size is too large. Please send an image smaller than 20 MB.")
        return

    try:
        file = await asyncio.wait_for(document.get_file(), timeout=2000)
    except asyncio.TimeoutError:
        await update.message.reply_text("Failed to download the file. It might be too large or the connection is unstable. Please try again later.")
        logger.error(f"Timeout downloading file from user {user_id}: {document.file_name}")
        return

    logger.info(f"Received file from user {user_id}: {document.file_name}")
    secure_original_filename = secure_filename(document.file_name)
    original_file_path = safe_join('uploads', secure_original_filename)
    await file.download_to_drive(original_file_path)
    print(f"File saved to: {original_file_path}")
    
    if document.mime_type.startswith('image/'):
        try:
            await process_and_send_image(update, context, original_file_path, document.file_name)
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            await update.message.reply_text("An error occurred while processing the image. Please try again or send a different image.")

## PROCESSING FUNCTION
async def process_and_send_image(update: Update, context: ContextTypes.DEFAULT_TYPE, original_file_path: str, file_name):
    """Processes the image, sends it, and handles session data."""
    try:
        user_id = update.effective_user.id
        user_name = escape_markdown(update.effective_user.first_name)
        session_id = str(uuid.uuid4())

        reply_markup = generate_main_keyboard(session_id)

        await update.message.reply_text(f"{user_name}, your image is being processed. Please wait...")
        await asyncio.sleep(5) 
        # Process the image
        image = cv2.imread(original_file_path)
        adjusted_image = MODELS[current_model].process_image(image)
        secure_adjusted_filename = secure_filename(f'adjusted_{file_name}')
        adjusted_file_path = os.path.join('uploads', secure_adjusted_filename)
        cv2.imwrite(adjusted_file_path, adjusted_image)

        # Remove the original image after processing
        if os.path.exists(original_file_path):
            os.remove(original_file_path)

        # Send the adjusted image with the keyboard attached
        with open(adjusted_file_path, 'rb') as adjusted_file:
            adjusted_msg = await context.bot.send_document(
                chat_id=update.message.chat_id,
                document=adjusted_file,
                caption=f"Here's the image with {current_model} adjustment applied, {user_name}.",
                reply_markup=reply_markup  # Attach keyboard here
            )
        
        context.user_data.setdefault('image_sessions', {})[session_id] = {
            'adjusted': [adjusted_file_path]
        }

        # Store file for cleanup (adjusted only)
        context.user_data.setdefault('files_to_delete', []).append({
            'path': adjusted_file_path,
            'timestamp': time.time(),
            'session_id': session_id
        })

    except Exception as e:
        logger.error(f"Error processing and sending image: {e}")
        await update.message.reply_text("An error occurred while processing the image. Please try again or send a different type of image.")

file_lock = asyncio.Lock()

## SECONDARY MODELS 
async def apply_more_models(action, session_id, update, context):
    """Applies additional models to the image."""
    logger.info(f"Applying model: {action}, Session ID: {session_id}")
    session_info = context.user_data.get('image_sessions', {}).get(session_id)
    
    if not session_info:
        await update.callback_query.message.reply_text("Session expired or invalid.")
        logger.error(f"Session not found for session_id: {session_id}")
        return

    try:
        original_file_path = session_info['adjusted'][-1]  # Get the most recent adjusted file path
        logger.info(f"Original file path: {original_file_path}")
        await asyncio.sleep(5)
        
        if os.path.exists(original_file_path):
            image = cv2.imread(original_file_path)
            if action == "contrast":
                processed = MODELS["contrast"].process_image(image)
                model_name = "contrast"
            elif action == "bw":
                processed = MODELS["bw"].process_image(image)
                model_name = "black and white"
            elif action == "upscale":
                processed = MODELS["upscale"].process_image(image)
                model_name = "upscale"
            elif action == "downscale":
                processed = MODELS["downscale"].process_image(image)
                model_name = "downscale"
            elif action == "darker":
                processed = MODELS["darker"].process_image(image)
                model_name = "darker enhance"
            elif action == "local contrast":
                processed = MODELS["local contrast"].process_image(image)
                model_name = "local contrast"
            else:
                raise ValueError("Invalid action")

            ext = os.path.splitext(original_file_path)[1] or '.jpg'
            processed_filename = f"{uuid.uuid4().hex}_{action}{ext}"
            processed_path = os.path.join('uploads', processed_filename)

            cv2.imwrite(processed_path, processed)

            reply_markup = generate_main_keyboard(session_id)

            with open(processed_path, "rb") as processed_file:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=processed_file,
                    caption=f"Here's the image with {model_name} adjustment applied.",
                    reply_markup=reply_markup
                )

            # Remove the old file from disk and session tracking
            if os.path.exists(original_file_path):
                os.remove(original_file_path)
            if original_file_path in session_info['adjusted']:
                session_info['adjusted'].remove(original_file_path)
            context.user_data['files_to_delete'] = [
                f for f in context.user_data.get('files_to_delete', [])
                if f['path'] != original_file_path
            ]

            # Track the new file
            session_info['adjusted'].append(processed_path)
            context.user_data['image_sessions'][session_id] = session_info
            context.user_data.setdefault('files_to_delete', []).append({
                'path': processed_path,
                'timestamp': time.time(),
                'session_id': session_id
            })

        else:
            logger.error(f"File not found: {original_file_path}")
            await update.callback_query.message.reply_text("The adjusted image is no longer available.")
            await update.callback_query.edit_message_reply_markup(reply_markup=None)

    except Exception as e:
        logger.exception(f"Unexpected error in apply_more_models: {e}")
        await update.callback_query.message.reply_text("An unexpected error occurred. Please try again later.")

# ASYNC FUNCTIONS
async def process_image_async(image, model):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, model.process_image, image)

## ACTIONS
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query

    # Block spamming: if already processing, ignore
    if context.user_data.get('processing', False):
        try:
            await query.answer(text="Please wait, still processing...", show_alert=False)
        except telegram.error.BadRequest:
            pass
        return

    try:
        await query.answer()
    except telegram.error.BadRequest:
        return

    logger.info(f"Received callback data: {query.data}")

    try:
        action, session_id = parse_callback_data(query.data)
    except ValueError:
        await query.message.reply_text("Invalid callback data format.")
        logger.error(f"Invalid callback data format: {query.data}")
        return

    user_id = update.effective_user.id

    session_info = context.user_data.get('image_sessions', {}).get(session_id)

    if not session_info:
        await query.message.reply_text("This image session has expired or is invalid.")
        logger.error(f"Session not found for session_id: {session_id}")
        return

    if action == 'remove':
        await confirm_delete(update, context, session_id)
    elif action == 'confirm_delete':
        await remove_file(update, context, session_id)
    elif action == 'cancel':
        await query.message.reply_text('Delete action canceled.')

### ADD ACTIONS HERE
    elif action in ['contrast', 'bw', 'upscale', 'downscale', 'darker', 'local contrast']:
        context.user_data['processing'] = True
        # Remove buttons while processing
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except telegram.error.BadRequest:
            pass
        await query.message.reply_text("Editing...")
        try:
            await apply_more_models(action, session_id, update, context)
        finally:
            context.user_data['processing'] = False

async def remove_all_files(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /remove command — removes all files for the user."""
    sessions = context.user_data.get('image_sessions', {})
    if not sessions:
        await update.message.reply_text("You have no files to remove.")
        return

    count = 0
    for session_info in sessions.values():
        for file_path in session_info.get('adjusted', []):
            if os.path.exists(file_path):
                os.remove(file_path)
                count += 1

    context.user_data['image_sessions'] = {}
    context.user_data['files_to_delete'] = []
    await update.message.reply_text(f"Removed {count} file(s) from the system.")
    logger.info(f"User {update.effective_user.id} removed {count} files via /remove")

async def remove_file(update: Update, context: ContextTypes.DEFAULT_TYPE, session_id: str) -> None:
    """Removes all files for the user (called from button callback)."""
    sessions = context.user_data.get('image_sessions', {})
    if not sessions:
        await update.callback_query.message.reply_text('No files to remove.')
        return

    count = 0
    for session_info in sessions.values():
        for file_path in session_info.get('adjusted', []):
            if os.path.exists(file_path):
                os.remove(file_path)
                count += 1

    context.user_data['image_sessions'] = {}
    context.user_data['files_to_delete'] = []
    await update.callback_query.message.reply_text(f'Removed {count} file(s) from the system.')

async def confirm_delete(update: Update, context: ContextTypes.DEFAULT_TYPE, session_id: str) -> None:
    """Sends a confirmation message to the user."""
    keyboard = [
        [InlineKeyboardButton("Delete my files.", callback_data=generate_callback_data("confirm_delete", session_id))],
        [InlineKeyboardButton("Keep editing", callback_data=generate_callback_data("cancel", session_id))]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.message.reply_text('Are you sure you want to delete the image?', reply_markup=reply_markup)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles errors during update processing."""

    logger.error("Exception while handling an update:", exc_info=context.error)

    if update and update.effective_message:
        await update.effective_message.reply_text(
            "An unexpected error occurred. Please try again later or contact support."
        )

# SCHEDULE FUNCTION
def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)

def main() -> None:
    try:
        developer_password = input("Enter developer password: ")
        if developer_password != DEVELOPER_PASSWORD:
            print("Incorrect password. Exiting...")
            sys.exit(1)

        print("Password verified. Starting the bot...")
        
        print("Initializing the database...")
        init_db()
        print("Database initialized successfully.")

        print("Setting up the uploads directory...")
        os.makedirs('uploads', exist_ok=True)
        print("Uploads directory is ready.")

        print("Building the bot application...")
        application = Application.builder().token(TOKEN).build()
        print("Bot application built successfully.")

        print("Adding command handlers...")
        application.add_handler(CommandHandler('start', start))
        application.add_handler(CommandHandler('remove', remove_all_files))
        application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
        application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        application.add_handler(CallbackQueryHandler(button_callback))
        print("Command handlers added.")

        print("Adding error handler...")
        application.add_error_handler(error_handler)
        print("Error handler added.")

        print("Scheduling cleanup job...")
        schedule.every(1).hour.do(cleanup_old_files)
        print("Cleanup job scheduled to run every hour.")

        print("Starting cleanup thread...")
        cleanup_thread = threading.Thread(target=run_schedule, daemon=True)
        cleanup_thread.start()
        print("Cleanup thread started.")

        logger.info("Bot started...")
        print("Bot is now running. Waiting for events...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    print("Starting the bot...")
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
