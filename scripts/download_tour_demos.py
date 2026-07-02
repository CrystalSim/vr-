from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


DEMOS = {
    "flores": {
        "url": "https://archive.org/download/coriaca-Flores_Park_360_Walking_Tour_1020_W_Etiwanda_Avenue_Rialto_CA/Flores_Park_360_Walking_Tour_1020_W_Etiwanda_Avenue_Rialto_CA.mp4",
        "filename": "flores_park_360_walking_tour.mp4",
    },
    "woodbury": {
        "url": "https://archive.org/download/cowomn-Woodbury_360_-_Central_Park/Woodbury_360_-_Central_Park.mp4",
        "filename": "woodbury_central_park_360.mp4",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download tour-guide 360 video demos.")
    parser.add_argument(
        "--out-dir",
        default="data/raw/videos/tour_scenarios",
        help="Directory where demo videos will be saved.",
    )
    parser.add_argument(
        "--demo",
        choices=[*DEMOS.keys(), "all"],
        default="all",
        help="Which demo to download.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = DEMOS if args.demo == "all" else {args.demo: DEMOS[args.demo]}

    for name, item in selected.items():
        target = out_dir / item["filename"]
        if target.exists() and target.stat().st_size > 0:
            print(f"Skip {name}: {target} already exists")
            continue
        print(f"Download {name}: {item['url']}")
        urllib.request.urlretrieve(item["url"], target)
        print(f"Saved {target} ({target.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
