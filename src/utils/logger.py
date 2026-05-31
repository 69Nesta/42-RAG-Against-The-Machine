from .color import Color

from pydantic import ValidationError
import datetime


class Logger:
    """Lightweight logger for console output.

    Attributes:
        print_log (bool): Whether logging is enabled.
        name (str): Name displayed in log messages.
        color (Color): Color used for the name tag.
    """
    verbose: bool
    name: str
    color: Color

    def __init__(
                self,
                name: str = 'Logger',
                color: Color = Color.GRAY,
                verbose: bool = False,
            ) -> None:
        """Initialize a logger instance.

        Args:
            verbose: Whether to enable logging output. Defaults to False.
            name: Display name for log messages. Defaults to 'Logger'.
            color: ANSI color for the name tag. Defaults to Color.GRAY.
        """
        self.verbose = verbose
        self.name = name
        self.color = color

    def log(self, message: str, end: str | None = '\n') -> None:
        """Print a debug/info message when logging is enabled.

        Args:
            message (str): Message to print.
            end (str | None): End character appended to the message.
        """
        if (self.verbose):
            print(f'{self._get_format()} {message}', end=end)

    def error(self, message: str, end: str | None = '\n') -> None:
        """Print an error message (always shown).

        Args:
            message (str): Error message to print.
            end (str | None): End character appended to the message.
        """
        print(
            f'{self._get_format()} {Color.RED}[ERROR]{Color.RESET} {message}',
            end=end
        )

    def warning(self, message: str, end: str | None = '\n') -> None:
        """Print a warning message (always shown).

        Args:
            message (str): Warning message to print.
            end (str | None): End character appended to the message.
        """
        print(
            f'{self._get_format()} {Color.YELLOW}[WARNING]{Color.RESET} '
            f'{message}',
            end=end
        )

    @staticmethod
    def warning_static(
                name: str,
                message: str,
                color: Color = Color.WHITE,
                end: str | None = '\n'
            ) -> None:
        """Print a warning message from a static context (always shown).

        Args:
            name (str): Name to include in the log prefix.
            message (str): Warning message to print.
            color (Color): Color to use for the name tag in the log prefix.
            end (str | None): End character appended to the message.
        """
        print(
            f'{Logger.get_format_static(color, name)} {Color.YELLOW}[WARNING]'
            f'{Color.RESET} {message}',
            end=end
        )

    def info(self, message: str, end: str | None = '\n') -> None:
        """Print an informational message (always shown).

        Args:
            message (str): Message to print.
            end (str | None): End character appended to the message.
        """
        print(
            f'{self._get_format()} [{Color.BRIGHT_CYAN}INFO{Color.RESET}] '
            f'{message}',
            end=end
        )

    def _get_format(self) -> str:
        """Return the formatted prefix used for all log lines.

        Returns:
            str: Formatted prefix including time and colored name tag.
        """
        return self.get_format_static(self.color, self.name)

    @staticmethod
    def get_format_static(color: Color, name: str) -> str:
        """Return the formatted prefix for static contexts.

        Args:
            name (str): Name to include in the prefix.
        Returns:
            str: Formatted prefix including time and colored name tag.
        """
        return f'{Color.GRAY}[{Logger.get_date_time()}] {color}[' +\
               f'{name}]{Color.RESET}'

    @staticmethod
    def get_date_time() -> str:
        """Return the current time string used in the log prefix.

        Returns:
            str: Formatted time string (HH:MM:SS).
        """
        now = datetime.datetime.now()
        return now.strftime("%X")

    def pydantic_error(self, e: ValidationError, message: str = '') -> None:
        """Print detailed error messages from a Pydantic ValidationError.

        Args:
            e (ValidationError): The ValidationError instance to process.
            message (str): Optional custom message to display before error
            details.
        """
        if message:
            self.error(message)

        for error in e.errors():
            loc: tuple[int | str, ...] = error.get('loc') or ('',)
            field = str(loc[0]) if loc else ''
            ctx = error.get('ctx') or {}

            if ctx_error := ctx.get('error'):
                self.error(f'Error: {ctx_error}')
            elif msg := error.get('msg'):
                if field:
                    self.error(f'Error: {field}: {msg}')
                else:
                    self.error(f'Error: {msg}')
