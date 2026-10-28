#!/usr/bin/env python3
"""Build private contact-sheet previews for human basketball frame triage.

No network calls. No confirmed training labels are ever generated. Contact
sheets contain identifiable court imagery: keep them under gitignored
private_training_data, never upload them to a public repository.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def build_sheets(session: Path, output: Path, columns=5, rows=6, thumb_size=(240, 150)):
    if columns < 1 or rows < 1 or columns * rows > 120:
        raise ValueError("Invalid contact-sheet grid")
    queue_file = session / "review_queue.csv"
    index_file = session / "index.csv"
    if not queue_file.is_file() or not index_file.is_file():
        raise FileNotFoundError("Need the local ingested session and review_queue.csv")
    with index_file.open(newline="", encoding="utf-8") as f:
        known = {row["filename"] for row in csv.DictReader(f)}
    with queue_file.open(newline="", encoding="utf-8") as f:
        selected = list(csv.DictReader(f))
    if not selected:
        raise ValueError("Empty review queue")
    filenames = [row["filename"] for row in selected]
    if len(set(filenames)) != len(filenames):
        raise ValueError("Duplicate frame names in review queue")
    for name in filenames:
        p = Path(name)
        if name not in known or p.is_absolute() or p.parent != Path("images") or p.suffix.lower() != ".jpg":
            raise ValueError("Unsafe or unknown image in review queue: " + name)
        if not (session / p).is_file():
            raise FileNotFoundError(session / p)

    # Never silently replace previously generated private work.
    if output.exists():
        raise FileExistsError("Refusing to overwrite contact sheets: " + str(output))
    output.mkdir(parents=True)
    page_size = columns * rows
    cell_w, cell_h = thumb_size[0], thumb_size[1] + 32
    pages = math.ceil(len(selected) / page_size)
    try:
        for page in range(pages):
            sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "#151515")
            draw = ImageDraw.Draw(sheet)
            subset = selected[page * page_size:(page + 1) * page_size]
            for index, frame in enumerate(subset):
                source = session / frame["filename"]
                with Image.open(source) as im:
                    im.load()
                    if im.width < 1 or im.height < 1 or im.width > 8192 or im.height > 8192:
                        raise ValueError("Invalid source image dimensions")
                    thumb = ImageOps.contain(im.convert("RGB"), thumb_size)
                x = (index % columns) * cell_w
                y = (index // columns) * cell_h
                sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y))
                draw.text((x + 5, y + thumb_size[1] + 4),
                          f"{page * page_size + index + 1}: {Path(frame['filename']).stem[:22]}",
                          fill="#FFFFFF")
            sheet.save(output / f"sheet_{page + 1:03d}.jpg", format="JPEG", quality=83)
        # A metadata-only index, matching the order of the displayed photos.
        (output / "contact_sheet_index.json").write_text(
            json.dumps({"session": session.name, "frames": filenames,
                        "pages": pages, "columns": columns, "rows": rows,
                        "labels_generated": 0}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception:
        for file in output.glob("*"):
            file.unlink()
        output.rmdir()
        raise
    return {"session": session.name, "frames": len(filenames), "sheets": pages,
            "output": str(output), "labels_generated": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="PRIVATE, gitignored directory for sheet JPEGs")
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--rows", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(build_sheets(args.session, args.output, args.columns, args.rows),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
