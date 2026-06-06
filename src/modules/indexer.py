import time

from ..interfaces import ChromaDBInterface, ChunksInterface, Bm25sInterface
from ..models import ChunkContentModel, MinimalSource
from ..utils import Logger, Color
from ..enums import IndexType
from ..config import Config

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from tqdm import tqdm
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from pathlib import Path


class IndexerConfig(BaseModel):
    root_path: str
    maximum_chunk_size: int = Field(..., gt=0, le=2000)
    index_type: IndexType

    overlap: int = Field(...,  gt=0, lt=100)


class IndexerModule:
    logger: Logger
    app_config: Config
    chromadb_interface: ChromaDBInterface
    chunks_interface: ChunksInterface
    bm25s_interface: Bm25sInterface

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
                overlap: int,
                chromadb_interface: ChromaDBInterface,
                chunks_interface: ChunksInterface,
                bm25s_interface: Bm25sInterface,
                config: Config,
            ) -> None:
        self.app_config = config
        self.chromadb_interface = chromadb_interface
        self.chunks_interface = chunks_interface
        self.bm25s_interface = bm25s_interface

        self.logger = Logger('IndexerModule', Color.CYAN, config.verbose)
        self.logger.log('Initializing Indexer Module...')

        self.config = IndexerConfig(
            root_path=root_path,
            maximum_chunk_size=maximum_chunk_size,
            index_type=index_type,
            overlap=overlap
        )

    def index(self) -> None:
        start_time: float = time.time()

        self._explore()
        self._initalize_splitters()
        self._create_config_files()
        self._index_files()

        self.logger.info(
            f'Indexing completed in {time.time() - start_time:.2f} seconds !'
        )

    def _explore(self) -> None:
        path: Path = Path(self.config.root_path)
        self.files_path = []
        self.logger.log(
            f'Exploring {path!r}...'
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
        chunk_overlap: int = chunk_size * (self.config.overlap // 100)

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
        chunks: dict[str, ChunkContentModel] = {}

        for file in tqdm(
            self.files_path,
            desc='Processing files',
            unit='file',
            disable=not self.app_config.verbose
        ):

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

            for chunk in tqdm(
                file_chunks,
                desc=f'Chunks from {file.name}',
                unit='chunk',
                leave=False,
            ):
                start_index: int = chunk.metadata['start_index']
                end_index: int = start_index + len(chunk.page_content)
                current_id: int = len(corpus)
                corpus_metadata[current_id] = {
                    'file_path': file_path_str,
                    'first_character_index': start_index,
                    'last_character_index': end_index,
                }

                chunk.metadata.update({
                    'first_character_index': start_index,
                    'last_character_index': end_index
                })
                del chunk.metadata['start_index']

                ids.append(str(current_id))

                chunks.update({
                    str(current_id): ChunkContentModel(
                        id=str(current_id),
                        content=chunk.page_content,
                        metadata=MinimalSource(**chunk.metadata)
                    )
                })

                file_path_formatted: str = file_path_str\
                    .replace('/', ' ')\
                    .replace('\\', ' ')\
                    .replace('.', ' ')\
                    .replace('_', ' ')\
                    .replace('-', ' ')

                bms25_txt = (
                    f'{chunk.page_content} {file_path_str*10} '
                    f'{file_path_formatted}'
                )
                corpus.append(bms25_txt)
            self.logger.log_tqdm(
                f'Indexed {len(file_chunks):3} chunks from {file_path_str!r} !'
            )

        self.logger.log(
            f'Indexed {len(corpus)} chunks from {len(self.files_path)} files !'
        )

        self.logger.log('Creating BM25 index...')

        self.bm25s_interface.index(corpus)
        self.bm25s_interface.save()

        self.chunks_interface.save_chunks(chunks)
        self.logger.info(
            'Saved BM25 index and metadata successfully !'
        )
        if self.app_config.use_chroma:
            self.logger.log('Creating ChromaDB index...')

            pbar = tqdm(
                total=len(ids),
                desc='Saving to ChromaDB',
                unit='chunk',
            )

            def progress_bar_func(current: int, _: int) -> None:
                pbar.update(current - pbar.n)

            self.chromadb_interface.batch_add(
                collection=self.chromadb_interface.get_collection(),
                ids=ids,
                documents=corpus,
                metadatas=[val for val in corpus_metadata.values()],
                progress_bar_func=progress_bar_func
            )
            pbar.close()
            self.logger.info(
                'Saved ChromaDB index successfully !'
            )

    def _create_config_files(self) -> None:
        folders: list[str] = [
            self.app_config.processed_bm25_index_path,
        ]
        files: list[str] = [
            self.app_config.processed_chunks_path
        ]

        for path in folders:
            Path(path).mkdir(parents=True, exist_ok=True)
        for path in files:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
