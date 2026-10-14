import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import shutil

from config.config import PipelineConfig


def reset(config: PipelineConfig) -> None:
    output_dir = config.storage.output_dir
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
        print(f"[reset] deleted {output_dir}")
    else:
        print(f"[reset] {output_dir} does not exist - nothing to delete")
    print("[reset] done (the send buffer is in memory only, so there is no database to wipe)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wipe the on-disk crop output")
    parser.add_argument("--config", help="Optional extra config file applied last (see config/config.py)")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    config = PipelineConfig.load(args.config)

    if not args.yes:
        answer = input(f"This will permanently delete {config.storage.output_dir}/ . Continue? [y/N] ")
        if answer.strip().lower() != "y":
            print("Cancelled - nothing was deleted.")
            return

    reset(config)


if __name__ == "__main__":
    main()
