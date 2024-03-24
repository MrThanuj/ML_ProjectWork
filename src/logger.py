import logging
import os
from datetime import datetime

# Create a log file name with the current timestamp
log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Define the directory where logs will be stored
log_directory = os.path.join(os.path.dirname(__file__), "application_logs")

# Ensure the directory exists; if not, create it
os.makedirs(log_directory, exist_ok=True)

# Construct the full path for the log file
full_log_path = os.path.join(log_directory, log_filename)

# Configure the logging
logging.basicConfig(
    filename=full_log_path,
    filemode='a',  # Append mode
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# Expose a custom log function for external use
def custom_log(message, level=logging.INFO):
    if level == logging.ERROR:
        logging.error(message)
    else:
        logging.info(message)
