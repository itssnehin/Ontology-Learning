import logging
from .config import LOGS_DIR

def setup_logging():
    """
    Sets up the root logger configuration.
    This is now handled centrally in config.py, but this function can be kept
    for modules that might need to ensure logging is configured.
    """
    # Configuration is now done in config.py to avoid multiple handlers.
    # This function can be used for module-specific logger setup if needed.
    pass