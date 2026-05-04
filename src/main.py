from .utils import Logger, Color


class RAG:
    logger: Logger

    def __init__(self, verbose: bool = False) -> None:
        self.logger = Logger('Main', Color.MAGENTA, verbose)
        self.logger.log('Initializing RAG...')

    def index(
                self,
                lib_path: str = 'data/raw/vllm-0.10.1',
                maximum_chunk_size: int = 5000
            ) -> None:
        from .Indexer import Indexer
        self.logger.log('Starting Indexer...')
        Indexer(lib_path, maximum_chunk_size)

    def search(self) -> None:
        self.logger.info('Not implemented')

    def search_dataset(self) -> None:
        self.logger.info('Not implemented')

    def answer(self) -> None:
        self.logger.info('Not implemented')

    def answer_dataset(self) -> None:
        self.logger.info('Not implemented')

    def evaluate(self) -> None:
        self.logger.info('Not implemented')
