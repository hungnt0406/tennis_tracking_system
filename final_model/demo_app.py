"""Gradio demo for the tennis analysis pipeline (ball + court + bounce).

Launch from the repo root with:

    python final_model/demo_app.py

Note: because this is run as a top-level script (not `python -m`), the
package-relative `from .pipeline import TennisPipeline` fails. The fallback
below adds the repo root to sys.path and imports via the full package path.
"""

import os
import sys
import tempfile

import gradio as gr

try:
    from .pipeline import TennisPipeline
except ImportError:
    # Running as a top-level script: add repo root to sys.path and import by full path.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from final_model.pipeline import TennisPipeline


def _run(video_path, bounds):
    pipeline = TennisPipeline(bounds=bounds)  # instantiate per-call (stateless), CPU default
    out_path = os.path.join(tempfile.mkdtemp(), "annotated.mp4")
    summary = pipeline.run(video_path, out_path, bounds=bounds)
    return out_path, summary  # (video, json) tuple


demo = gr.Interface(
    fn=_run,
    inputs=[
        gr.Video(label="Upload clip"),
        gr.Radio(["singles", "doubles"], value="singles", label="Court bounds"),
    ],
    outputs=[
        gr.Video(label="Annotated output"),
        gr.JSON(label="Summary"),
    ],
    title="Tennis Analysis — ball + court + bounce",
    description="Upload a clip to track the ball, detect the court, and find bounces. In/out calls are qualitative.",
)


if __name__ == "__main__":
    demo.launch()
