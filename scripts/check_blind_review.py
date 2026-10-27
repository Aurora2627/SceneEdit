"""Verify a generated blind package without opening its private answer key."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scenedit.common import read, sha
from scenedit.review import protocol_id


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.videos = []
        self.selects = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "video":
            self.videos.append(attrs.get("src"))
        if tag == "select":
            self.selects += 1


def verify(out):
    document = read(out / "tasks.json")
    if protocol_id(document["tasks"]) != document["protocol_id"]:
        raise ValueError("Changed task manifest")
    for task in document["tasks"]:
        if set(task["assets"]) != {"source", "A", "B"}:
            raise ValueError("Missing video asset")
        for label, relative in task["assets"].items():
            path = (out / relative).resolve()
            if (
                not path.is_relative_to(out.resolve())
                or sha(path) != task["sha256"][label]
            ):
                raise ValueError("Changed or escaped video asset")
    page = Page()
    page.feed((out / "index.html").read_text(encoding="utf-8"))
    expected = [
        task["assets"][label]
        for task in document["tasks"]
        for label in ("source", "A", "B")
    ]
    if page.videos != expected or page.selects != 5 * len(document["tasks"]):
        raise ValueError("Page and manifest disagree")
    if list(out.glob("*key*")):
        raise ValueError("Private answer key was placed inside the shared package")
    print(
        "BLIND_PACKAGE_VERIFIED",
        len(document["tasks"]),
        "tasks",
        len(page.videos),
        "videos",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    verify(parser.parse_args().package)
