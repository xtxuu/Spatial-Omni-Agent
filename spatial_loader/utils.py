"""Small filesystem/serialization helpers, with paths relative to a release root."""
import hashlib
import json
import time
from pathlib import Path

from .contracts import DataError


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.part')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    for attempt in range(8):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 7: raise
            time.sleep(min(.05 * 2**attempt,.4))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve(root, relative):
    root=Path(root).resolve();p=(root/relative).resolve()
    if Path(relative).is_absolute() or not p.is_relative_to(root):
        raise DataError('UNSAFE_PATH', f'Path is outside release: {relative}')
    return p


def sql_string(value):
    # Only internally resolved filesystem paths use SQL literals; filters use parameters.
    return "'"+str(value).replace("'","''")+"'"
