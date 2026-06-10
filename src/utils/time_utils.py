from time import time


class TimeUtils:
    start_time: float

    def __init__(self) -> None:
        self.start()

    def start(self) -> None:
        self.start_time = time()

    def get_elapsed_time(self) -> float:
        return time() - self.start_time

    def get_elapsed_time_formated(self) -> str:
        elapsed: int = int(self.get_elapsed_time())

        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours:
            return f'{hours}h {minutes}min {seconds}s'
        if minutes:
            return f'{minutes}min {seconds}s'
        return f'{seconds}s'
