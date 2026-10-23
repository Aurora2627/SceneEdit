"""Build labeled contact sheets from unchanged input and generated frames."""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from scenedit.common import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    for case in ("car-roundabout", "car-turn"):
        prepared = args.base / case / "prepared-pilot"
        meta = read(prepared / "prepared.json")
        profile = meta["profile"]
        indices = [0, profile["num_frames"] // 2, profile["num_frames"] - 1]
        rows = [
            ("source", prepared / "source"),
            ("edit-mask (white=edit)", prepared / "edit-mask"),
        ]
        for mode in ("with-reference", "without-reference"):
            run = args.base / "baseline-v1" / (case + "-" + mode)
            if (run / "status.json").exists() and read(run / "status.json").get(
                "status"
            ) == "complete":
                rows.append((mode, run / "frames"))
        width, height = profile["width"], profile["height"]
        sheet = Image.new("RGB", (width * 3, (height + 24) * len(rows)), (35, 35, 35))
        draw = ImageDraw.Draw(sheet)
        for row, (label, folder) in enumerate(rows):
            for col, index in enumerate(indices):
                image = Image.open(folder / (str(index).zfill(4) + ".png")).convert(
                    "RGB"
                )
                sheet.paste(image, (col * width, row * (height + 24) + 24))
                draw.text(
                    (col * width + 6, row * (height + 24) + 5),
                    label + " | frame " + str(index),
                    fill="white",
                )
        output = args.base / "baseline-v1" / (case + "-review.jpg")
        sheet.save(output, quality=95)
        print("REVIEW_SHEET", output, len(rows))


if __name__ == "__main__":
    main()
