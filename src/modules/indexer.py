from ..utils import Logger, Color
from ..enums import IndexType

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from pathlib import Path


class Indexer:
    logger: Logger

    # Config
    root_path: str
    maximum_chunk_size: int
    index_type: IndexType

    ALLOWED_FILES: dict[IndexType, set[str]] = {
        IndexType.CODE: { '.py' },
        IndexType.DOCS: { '.md', '.toml', '.txt' },
        IndexType.ALL: { '.py', '.md', '.toml', '.txt' }
    }

    files_path: list[Path]

    files_splitter: dict[str, RecursiveCharacterTextSplitter]
    default_splitter: RecursiveCharacterTextSplitter

    def __init__(self, root_path: str, maximum_chunk_size: int, index_type: IndexType, verbose: bool) -> None:
        self.logger = Logger('Indexer', verbose, Color.CYAN)
        self.logger.log('Initializing Indexer...')

        # Set config
        self.root_path = root_path
        if maximum_chunk_size <= 0:
            raise ValueError('Maximum chunk size must be a positive integer')
        elif maximum_chunk_size > 2000:
            raise ValueError('Maximum chunk size must be less than or equal to 2000')
        try:
            self.maximum_chunk_size = int(maximum_chunk_size)
        except ValueError as e:
            raise ValueError(f'Invalid maximum chunk size: {e}')
        try:
            self.index_type = IndexType(index_type)
        except ValueError as e:
            raise ValueError(f'Invalid index type: {e}. Allowed values are: {[t.value for t in IndexType]}')

        self._explore()
        self._initalize_text_splitter()
        self._index_code_files()
        # self._index_docs_files()

    def _explore(self) -> None:
        path: Path = Path(self.root_path)
        self.files_path = []
        self.logger.log(
            f'Exploring {path}...'
        )

        for file in path.rglob('*'):
            if not file.is_file():
                continue
            if file.suffix.lower() in self.ALLOWED_FILES.get(self.index_type, {}):
                self.files_path.append(file)

        self.logger.log(
            f'Found {len(self.files_path)} files !'
        )

    def _initalize_text_splitter(self) -> None:
        self.files_splitter = {
            '.py': RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.maximum_chunk_size,
                chunk_overlap=self.maximum_chunk_size * 5 // 100,
                add_start_index=True,
            ),
            '.md': RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=self.maximum_chunk_size,
                chunk_overlap=self.maximum_chunk_size * 5 // 100,
                add_start_index=True
            )
        }
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.maximum_chunk_size,
            chunk_overlap=self.maximum_chunk_size * 5 // 100,
            add_start_index=True,
        )

    def _index_code_files(self) -> None:
        self.logger.log('Indexing code files...')
        # file = self.files_path[9]

        for file in self.files_path:
            self.logger.log(f'Indexing {file}...')
            content: str = ''
            try:
                with open(file, "r") as f:
                    content = f.read()
            except UnicodeDecodeError:
                continue

            chunks: list[Document] = self.files_splitter.get(
                file.suffix,
                self.default_splitter
            ).create_documents(
                [content],
                metadatas=[{
                    "source": file.as_posix()
                }]
            )

            self.logger.log(f'Indexed {len(chunks)} chunks from {file} !')

            for i, chunk in enumerate(chunks):
                end_index: int = chunk.metadata["start_index"]
                end_index += len(chunk.page_content)

                chunk.metadata.update({
                    'end_index': end_index
                })
                self.logger.log(f'Chunk {i}: {len(chunk.page_content)} characters, source: {chunk.metadata}')

    # def _index_docs_files(self) -> None:
    #     self.logger.log('Indexing docs files...')
    #     file = self.docs_files_path[0]
    #     # for file in self.docs_files_path:
    #     self.logger.log(f'Indexing {file}...')
    #     with open(file, 'r') as f:
    #         content = f.read()
    #     chunks = self.docs_splitter.split_text(content)
