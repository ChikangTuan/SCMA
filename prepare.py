import argparse
from pathlib import Path

from scma.data import prepare_case
from scma.geometry import FanBeam640


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    with FanBeam640() as projector:
        prepare_case(args.case, args.mask, args.output, projector)


if __name__ == "__main__":
    main()
