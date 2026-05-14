"""Display the latest saved checkpoint for a JSON experiment config."""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", help="Path to the JSON experiment config.")
    parser.add_argument(
        "--mode",
        choices=("auto", "layer", "toolpath"),
        default="auto",
        help="Display mode. Auto detects from checkpoints in the output folder.",
    )
    parser.add_argument("--layer-count", type=int, default=15, help="Number of layer contours to draw.")
    parser.add_argument("--curve-count", type=int, default=30, help="Target number of toolpath curves per layer.")
    parser.add_argument("--tube-radius", type=float, default=0.2, help="Toolpath tube radius.")
    parser.add_argument("--subdivide-layers", action="store_true", help="Apply one subdivision pass to layer surfaces.")
    parser.add_argument("--hide-platform", action="store_true", help="Do not draw the platform checkpoint.")
    parser.add_argument("--camera", default=None, help="PyVista camera position, such as xy, zy, or xz.")
    return parser.parse_args()


def main():
    args = parse_args()
    from checkpoint_display import display_latest_checkpoint

    display_latest_checkpoint(
        args.config_path,
        mode=args.mode,
        layer_count=args.layer_count,
        curve_count=args.curve_count,
        tube_radius=args.tube_radius,
        subdivide_layers=args.subdivide_layers,
        show_platform=not args.hide_platform,
        cpos=args.camera,
    )


if __name__ == "__main__":
    main()
