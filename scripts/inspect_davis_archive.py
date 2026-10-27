"""Record the pinned DAVIS archive inventory before choosing available sequences."""

from collections import Counter
import zipfile
from pathlib import Path
from http_range import RangeReader
from prepare_davis_cases import URL, ETAG, SIZE
from scenedit.common import save

if __name__ == "__main__":
    with RangeReader(URL, ETAG, SIZE) as remote, zipfile.ZipFile(remote) as archive:
        names = archive.namelist()
        counts = Counter(
            n.split("/")[3]
            for n in names
            if n.startswith("DAVIS/JPEGImages/480p/") and n.endswith(".jpg")
        )
        target = Path("D:/career-projects/data/SceneEdit/davis-inventory.json")
        save(
            target,
            dict(
                url=URL,
                etag=ETAG,
                bytes=SIZE,
                image_counts=dict(sorted(counts.items())),
            ),
        )
        print(
            "VEHICLE_SEQUENCES",
            {
                name: count
                for name, count in counts.items()
                if "car" in name or "drift" in name
            },
            flush=True,
        )
