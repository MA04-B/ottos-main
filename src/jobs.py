import threading
from concurrent.futures import ThreadPoolExecutor

from src.data.database import Database


class TrainingJob:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lottery-training")
        self.future = None
        self.message = "Idle"
        self.progress = 0.
        self.lock = threading.Lock()

    def update(self, *args, **kwargs):
        with self.lock:
            numeric = [arg for arg in args if isinstance(arg, (int, float))]
            for arg in args:
                if isinstance(arg, str):
                    self.message = arg
            if len(numeric) >= 2 and numeric[1] > 0:
                self.progress = max(0., min(1., float(numeric[0]) / numeric[1]))
            elif numeric:
                self.progress = max(0., min(1., float(numeric[0])))

    def start(self, frame, names, validation_draws, seed, training_window):
        with self.lock:
            if self.future and not self.future.done():
                raise RuntimeError("Training is already running")
            self.message, self.progress = "Preparing chronological features", 0.
            self.future = self.executor.submit(self.run, frame.copy(), names, validation_draws, seed, training_window)

    def run(self, frame, names, validation_draws, seed, training_window):
        from src.ml.training import train_experiment
        db = Database()
        db.log("INFO", "training_started", ", ".join(names))
        try:
            result = train_experiment(frame, names, validation_draws=validation_draws,
                                      seed=seed, progress=self.update, training_window=training_window)
            db.save_experiment(result, len(frame))
            self.message, self.progress = "Training complete", 1.
            db.log("INFO", "training_complete", result["selected_model"])
            return result
        except Exception as exc:
            db.log("ERROR", "training_failed", repr(exc))
            raise
