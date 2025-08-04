import logging
from pathlib import Path

def setup_logging(output_dir: str, log_name: str):
    """
    Set up logging to file and console.
    
    Args:
        output_dir: Directory to store log files.
        log_name: Name of the log file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Path(output_dir) / f"{log_name}.log"),
            logging.StreamHandler()
        ]
    )