from .logger_utils import Logger


class Loader:
    logger: Logger

    LOGO: list[str] = [
        '',
        '___________          _____                _____        ',
        '\\          \\       /      |_         _____\\    \\_      ',
        ' \\    /\\    \\     /         \\       /     /|     |     ',
        '  |   \\_\\    |   |     /\\    \\     /     / /____/|     ',
        '  |      ___/    |    |  |    \\   |     | |_____|/     ',
        '  |      \\  ____ |     \\/      \\  |     | |_________   ',
        ' /     /\\ \\/   \\ \\|\\      /\\     \\ |\\     \\|\\        \\  ',
        '/_____/ |\\______|| \\_____\\ \\_____\\| \\_____\\|    |\\__/| ',
        '|     | | |     || |     | |     || |     /____/| | || ',
        '|_____|/ \\|_____| \\|_____|\\|_____| \\|_____|     |\\|_|/ ',
        '                                          |____/       ',
        ''
    ]

    def __init__(self, logger: Logger) -> None:
        self.logger = logger

    def print_logo(self) -> None:
        for line in self.LOGO:
            self.logger.info('              ' + line)
