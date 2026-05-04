from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from pathlib import Path
from .utils import Logger, Color


class Indexer:
    logger: Logger

    # Config
    root_path: str
    maximum_chunk_size: int

    ALLOWED_DOCS_FILES: set[str] = {'.md', '.toml', '.txt'}
    ALLOWED_CODE_FILES: set[str] = {'.py'}

    code_files_path: list[Path]
    docs_files_path: list[Path]

    code_splitter: RecursiveCharacterTextSplitter

    def __init__(self, root_path: str, maximum_chunk_size: int) -> None:
        self.logger = Logger('Indexer', Color.CYAN, True)
        self.logger.log('Initializing Indexer...')

        # Set config
        self.root_path = root_path
        self.maximum_chunk_size = maximum_chunk_size

        self._explore()
        self._initalize_text_splitter()
        self._index_code_files()
        # self._index_docs_files()

    def _explore(self) -> None:
        path: Path = Path('data/raw/vllm-0.10.1/vllm/model_executor/model_loader/')
        self.code_files_path = []
        self.docs_files_path = []

        for file in path.rglob('*'):
            if file.suffix in self.ALLOWED_CODE_FILES:
                self.code_files_path.append(file)
            if file.suffix in self.ALLOWED_DOCS_FILES:
                self.docs_files_path.append(file)

        self.logger.log(
            f'Found {len(self.code_files_path)} code files and '
            f'{len(self.docs_files_path)} docs files !'
        )

    def _initalize_text_splitter(self) -> None:
        self.code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.maximum_chunk_size,
            chunk_overlap=self.maximum_chunk_size * 5 // 100,
            add_start_index=True,
        )

    def _index_code_files(self) -> None:
        self.logger.log('Indexing code files...')
        file = self.code_files_path[9]

        # for file in self.code_files_path:
        self.logger.log(f'Indexing {file}...')
        with open(file, 'r') as f:
            content = f.read()
        chunks = self.code_splitter.create_documents([content], metadatas=[{"source": file.as_posix()}])

        self.logger.log(f'Indexed {len(chunks)} chunks from {file} !')

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({'end_index': chunk.metadata["start_index"] + len(chunk.page_content)})
            self.logger.log(f'Chunk {i}: {len(chunk.page_content)} characters, source: {chunk.metadata}')

    # def _index_docs_files(self) -> None:
    #     self.logger.log('Indexing docs files...')
    #     file = self.docs_files_path[0]
    #     # for file in self.docs_files_path:
    #     self.logger.log(f'Indexing {file}...')
    #     with open(file, 'r') as f:
    #         content = f.read()
    #     chunks = self.docs_splitter.split_text(content)
