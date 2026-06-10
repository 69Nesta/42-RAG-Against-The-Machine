from .file_type import FileType

from typing import Mapping


FilesExt: Mapping[FileType, set[str]] = {
    FileType.CODE: {'.py'},
    FileType.DOCS: {'.md', '.toml', '.txt'},
    FileType.ALL: {'.md', '.toml', '.txt', '.py'},
}
