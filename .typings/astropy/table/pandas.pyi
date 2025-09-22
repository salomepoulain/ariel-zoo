from IPython import display
from _typeshed import Incomplete

ascii_coded: str
ascii_uncoded: Incomplete
url: str
message_coded: str
message_uncoded: Incomplete
html: Incomplete

class HTMLWithBackup(display.HTML):
    backup_text: Incomplete
    def __init__(self, data, backup_text) -> None: ...
    def __repr__(self) -> str: ...

dhtml: Incomplete
