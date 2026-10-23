"""Seekable HTTP Range reader; rejects full-file fallbacks and changed archives."""

import io, time
from collections import OrderedDict
import requests


class RangeReader(io.RawIOBase):
    def __init__(self, url, etag, expected_size, block_size=262144):
        self.url = url
        self.etag = etag
        self.size = expected_size
        self.position = 0
        self.block_size = block_size
        self.cache = OrderedDict()
        self.session = requests.Session()
        self.bytes_downloaded = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = (
            offset
            if whence == 0
            else (
                self.position + offset
                if whence == 1
                else self.size + offset if whence == 2 else -1
            )
        )
        if position < 0:
            raise ValueError("Invalid seek")
        self.position = position
        return position

    def block(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        for attempt in range(4):
            try:
                with self.session.get(
                    self.url,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "If-Match": self.etag,
                        "Accept-Encoding": "identity",
                    },
                    stream=True,
                    timeout=(30, 90),
                ) as r:
                    if r.status_code != 206:
                        raise RuntimeError(
                            "Server did not honor partial content: "
                            + str(r.status_code)
                        )
                    if r.headers.get("ETag") != self.etag:
                        raise RuntimeError("Archive identity changed")
                    if (
                        r.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{self.size}"
                    ):
                        raise RuntimeError("Wrong HTTP byte range")
                    data = b"".join(r.iter_content(65536))
                    if len(data) != end - start + 1:
                        raise RuntimeError("Incomplete HTTP byte range")
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        self.bytes_downloaded += len(data)
        self.cache[index] = data
        while len(self.cache) > 16:
            self.cache.popitem(last=False)
        return data

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, max(0, self.size - self.position))
        if size > 16 * 1024 * 1024:
            raise ValueError("Refusing oversized ZIP read")
        chunks = []
        while size:
            index = self.position // self.block_size
            offset = self.position % self.block_size
            block = self.block(index)
            n = min(size, len(block) - offset)
            chunks.append(block[offset : offset + n])
            self.position += n
            size -= n
        return b"".join(chunks)

    def close(self):
        self.session.close()
        super().close()
