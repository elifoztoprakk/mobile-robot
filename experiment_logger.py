import csv
import json
import os
import time
from datetime import datetime


class ExperimentLogger:
    def __init__(self, experiment_name, output_dir="experiment_logs", config=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{timestamp}"
        self.run_dir = os.path.join(output_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.summary_path = os.path.join(self.run_dir, "summary.json")
        self.steps_path = os.path.join(self.run_dir, "steps.csv")
        self.events_path = os.path.join(self.run_dir, "events.csv")

        self.config = config or {}
        self.start_time = time.time()

        self._step_file = open(self.steps_path, "w", newline="")
        self._event_file = open(self.events_path, "w", newline="")

        self.step_writer = None
        self.event_writer = csv.DictWriter(
            self._event_file,
            fieldnames=["step", "time", "event_type", "details"],
        )
        self.event_writer.writeheader()

    def log_step(self, row):
        if self.step_writer is None:
            self.step_writer = csv.DictWriter(self._step_file, fieldnames=list(row.keys()))
            self.step_writer.writeheader()

        self.step_writer.writerow(row)

    def log_event(self, step, sim_time, event_type, details):
        self.event_writer.writerow({
            "step": step,
            "time": sim_time,
            "event_type": event_type,
            "details": json.dumps(details),
        })

    def save_summary(self, summary):
        elapsed = time.time() - self.start_time

        payload = {
            "run_id": self.run_id,
            "elapsed_wall_time_seconds": elapsed,
            "config": self.config,
            "summary": summary,
        }

        with open(self.summary_path, "w") as f:
            json.dump(payload, f, indent=2)

    def close(self):
        self._step_file.close()
        self._event_file.close()