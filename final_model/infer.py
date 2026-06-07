"""CLI wrapper around TennisPipeline.

Usage:
    python -m final_model.infer Dataset/game1/Clip1 --output /tmp/out.mp4
    python -m final_model.infer match.mp4 --output /tmp/out.mp4 --bounds doubles --device cuda
"""
import argparse
import json

from .pipeline import TennisPipeline


def main():
    parser = argparse.ArgumentParser(description="Run the tennis tracking pipeline on a video or frame directory.")
    parser.add_argument("src", help="Video file or directory of frames.")
    parser.add_argument("--output", default="final_model_out.mp4")
    parser.add_argument("--bounds", choices=["singles", "doubles"], default="singles")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ball-decode", dest="ball_decode", choices=["hough", "argmax"], default="hough")
    parser.add_argument("--fps", type=float, default=25, help="Output FPS used when src is a frame directory.")
    parser.add_argument("--ball-ckpt", dest="ball_ckpt", default=None)
    parser.add_argument("--court-ckpt", dest="court_ckpt", default=None)
    parser.add_argument("--bounce-ckpt", dest="bounce_ckpt", default=None)
    args = parser.parse_args()

    pipeline = TennisPipeline(
        ball_ckpt=args.ball_ckpt,
        court_ckpt=args.court_ckpt,
        bounce_ckpt=args.bounce_ckpt,
        device=args.device,
        bounds=args.bounds,
        ball_decode=args.ball_decode,
        fps=args.fps,
    )
    summary = pipeline.run(args.src, args.output, bounds=args.bounds)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
