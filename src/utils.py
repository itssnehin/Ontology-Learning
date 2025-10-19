import logging
from pathlib import Path

import csv
from datetime import datetime
import threading

class CostLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance 

    def __init__(self, output_dir="data/integrated_output"):
        if self._initialized:
            return
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = self.output_dir / f"cost_log_{timestamp}.csv"
        
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'stage', 'model', 'input_tokens', 'output_tokens', 'cost'])
        
        self._initialized = True

    def log_cost(self, stage: str, model: str, input_tokens: int, output_tokens: int, cost: float):
        """Logs a cost entry to the CSV file in a thread-safe manner."""
        with self._lock:
            timestamp = datetime.now().isoformat()
            with open(self.filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, stage, model, input_tokens, output_tokens, f"{cost:.6f}"])

# Singleton instance
cost_logger = CostLogger()