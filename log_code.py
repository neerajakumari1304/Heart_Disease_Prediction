import logging
import os

def setup_logging(script_name="main"):
    log_dir = "D:\\Heart_disease_prediction\\pythonProject\\logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{script_name}.log")

    # Create a named logger (not the root)
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
