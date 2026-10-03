"""Upload validation shared by the API and offline regression tests."""
import re
from pathlib import Path

ALLOWED_SUFFIXES = {'.pdf', '.docx', '.csv', '.txt', '.md'}


def safe_upload_name(name):
    if not name or len(name) > 240 or '/' in name or '\\' in name:
        raise ValueError('Supply a plain filename without directory components.')
    if name in {'.', '..'} or re.search(r'[\x00-\x1f\x7f:]', name):
        raise ValueError('Filename contains unsafe characters.')
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError('Unsupported document extension.')
    return name


async def bounded_upload(upload, destination, max_bytes):
    total = 0
    with Path(destination).open('xb') as stream:
        while True:
            chunk = await upload.read(min(65536, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError('Document exceeds upload size limit.')
            stream.write(chunk)
    return total
