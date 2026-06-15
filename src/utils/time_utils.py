from time import time


class TimeUtils:
    '''
    A utility class for measuring elapsed time.
    '''
    start_time: float

    def __init__(self) -> None:
        '''
        Initializes the TimeUtils instance and starts the timer.
        '''
        self.start()

    def start(self) -> None:
        '''
        Starts or restarts the timer by recording the current time.
        '''
        self.start_time = time()

    def get_elapsed_time(self) -> float:
        '''
        Returns the elapsed time since the timer was started.
        '''
        return time() - self.start_time

    def get_elapsed_time_formated(self) -> str:
        '''
        Returns the elapsed time in a formatted string.
        '''
        elapsed: int = int(self.get_elapsed_time())

        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours:
            return f'{hours}h {minutes}min {seconds}s'
        if minutes:
            return f'{minutes}min {seconds}s'
        return f'{seconds}s'
