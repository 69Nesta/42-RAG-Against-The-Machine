from ..utils import Logger, Color
from ..config import Config


class Bm25sInterface:
    logger: Logger
    app_config: Config

    def __init__(self, config: Config):
        self.app_config = config
        self.logger = Logger('Bm25sInterface', Color.YELLOW, config.verbose)
