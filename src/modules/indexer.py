from ..interfaces import ChromaDBInterface
from ..utils import Logger, Color
from ..enums import IndexType
from ..utils import JSONUtils
from ..config import Config

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from pydantic import BaseModel, Field, model_validator
from pathlib import Path
import bm25s


class IndexerConfig(BaseModel):
    root_path: str
    maximum_chunk_size: int = Field(..., gt=0, le=2000)
    index_type: IndexType
    verbose: bool

    processed_bm25_index_path: str = Field(..., min_length=1)
    processed_chunks_path: str = Field(..., min_length=1)
    processed_chunks_metadata_path: str = Field(..., min_length=1)

    @model_validator(mode='after')
    def check_paths_differ(self) -> 'IndexerConfig':
        if self.processed_bm25_index_path == self.processed_chunks_path:
            raise ValueError(
                'processed_bm25_index_path and processed_chunks_path must '
                'be different!'
            )

        checks = [
            (self.processed_bm25_index_path, True),
            (self.processed_chunks_path, True),
            (self.processed_chunks_metadata_path, False),
        ]

        for path, should_be_dir in checks:
            p = Path(path)
            if p.exists() and p.is_dir() != should_be_dir:
                kind = "directory" if should_be_dir else "file"
                raise ValueError(f"{path} must be a {kind}!")

        return self


class Indexer:
    logger: Logger
    app_config: Config
    chromadb_interface: ChromaDBInterface

    config: IndexerConfig

    FILES_TYPES: dict[IndexType, set[str]] = {
        IndexType.CODE: {'.py'},
        IndexType.DOCS: {'.md', '.toml', '.txt'},
    }

    ALLOWED_FILES: dict[IndexType, set[str]] = {
        IndexType.CODE: FILES_TYPES.get(IndexType.CODE, set()),
        IndexType.DOCS: FILES_TYPES.get(IndexType.DOCS, set()),
    }

    files_path: list[Path]

    files_splitter: dict[str, RecursiveCharacterTextSplitter]
    default_splitter: RecursiveCharacterTextSplitter

    def __init__(
                self,
                root_path: str,
                maximum_chunk_size: int,
                index_type: IndexType,
                processed_bm25_index_path: str,
                processed_chunks_path: str,
                processed_chunks_metadata_path: str,
                chromadb_interface: ChromaDBInterface,
                config: Config,
            ) -> None:
        self.app_config = config
        self.chromadb_interface = chromadb_interface

        self.logger = Logger('Indexer', Color.CYAN, config.verbose)
        self.logger.log('Initializing Indexer...')

        self.config = IndexerConfig(
            root_path=root_path,
            maximum_chunk_size=maximum_chunk_size,
            index_type=index_type,
            verbose=config.verbose,
            processed_bm25_index_path=Path(
                str(processed_bm25_index_path)
            ).as_posix(),
            processed_chunks_path=Path(
                str(processed_chunks_path)
            ).as_posix(),
            processed_chunks_metadata_path=Path(
                str(processed_chunks_metadata_path)
            ).as_posix()
        )

        self._explore()
        self._initalize_splitters()
        self._create_config_files()
        self._index_files()

    def _explore(self) -> None:
        path: Path = Path(self.config.root_path)
        self.files_path = []
        self.logger.log(
            f'Exploring {path}...'
        )

        allowed_ext: set[str] = self.ALLOWED_FILES.get(
            self.config.index_type,
            set().union(*self.ALLOWED_FILES.values())
        )
        for file in path.rglob('*'):
            if not file.is_file():
                continue
            if file.suffix.lower() in allowed_ext:
                self.files_path.append(file)

        self.logger.info(
            f'Found {len(self.files_path)} files !'
        )

    def _initalize_splitters(self) -> None:
        chunk_size: int = self.config.maximum_chunk_size
        chunk_overlap: int = chunk_size * 5 // 100

        self.files_splitter = {
            '.py': RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
            ),
            '.md': RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True
            )
        }
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def _index_files(self) -> None:
        self.logger.log('Indexing code files...')
        ids: list[str] = []
        corpus: list[str] = []
        corpus_metadata: dict[int, dict[str, str | int]] = {}

        for file in self.files_path:
            self.logger.log(f'Indexing {file}...')

            file_path_str: str = file.as_posix()
            content: str = ''
            try:
                with open(file, 'r') as f:
                    content = f.read()
            except UnicodeDecodeError:
                self.logger.warning(f'Failed to decode {file}')
                continue

            file_chunks: list[Document] = self.files_splitter.get(
                file.suffix,
                self.default_splitter
            ).create_documents(
                [content],
                metadatas=[{
                    'file_path': file_path_str
                }]
            )

            for chunk in file_chunks:
                start_index: int = chunk.metadata['start_index']
                end_index: int = start_index + len(chunk.page_content)
                current_id: int = len(corpus)

                chunk.metadata.update({
                    'first_character_index': start_index,
                    'last_character_index': end_index
                })
                del chunk.metadata['start_index']

                ids.append(str(current_id))
                corpus_metadata[current_id] = {
                    'file_path': file_path_str,
                    'first_character_index': start_index,
                    'last_character_index': end_index,
                }
                corpus.append(chunk.page_content)
            self.logger.log(f'Indexed {len(file_chunks)} chunks from {file} !')

        self.logger.log(
            f'Indexed {len(corpus)} chunks from {len(self.files_path)} files !'
        )

        self.logger.log('Creating BM25 index...')
        corpus_tokens = bm25s.tokenize(corpus)
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)

        self.logger.log(
            f'Saving BM25 index to {self.config.processed_bm25_index_path}...'
        )
        retriever.save(self.config.processed_bm25_index_path)
        JSONUtils.save_json(
            corpus_metadata,
            self.config.processed_chunks_metadata_path
        )
        self.logger.log(
            'Saved BM25 index and metadata successfully !'
        )
        if self.app_config.use_chroma:
            self.logger.log('Creating ChromaDB index...')

            def progress_bar_func(current: int, total: int) -> None:
                self.logger.log(
                    f'Saving ChromaDB index... {current}/{total} chunks saved'
                )

            self.chromadb_interface.batch_add(
                collection=self.chromadb_interface.get_collection(),
                ids=ids,
                documents=corpus,
                metadatas=[val for val in corpus_metadata.values()],
                progress_bar_func=progress_bar_func
            )
            self.logger.log(
                'Saved ChromaDB index successfully !'
            )

    def _create_config_files(self) -> None:
        folders: list[str] = [
            self.config.processed_bm25_index_path,
            self.config.processed_chunks_path,
        ]
        files: list[str] = [
            self.config.processed_chunks_metadata_path
        ]

        for path in folders:
            Path(path).mkdir(parents=True, exist_ok=True)
        for path in files:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
