from __future__ import annotations

from pathlib import Path

from drivesafe.llm import load_prompt_template

REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTION_PROMPT_PATH = REPO_ROOT / "configs" / "caption_prompt.txt"


def build_caption_messages(
    spatial_context: str,
    motion_context: str,
    depth_context: str,
    video_description: str,
    prompt_path: Path | None = None,
) -> list[dict[str, str]]:
    template = load_prompt_template(prompt_path or CAPTION_PROMPT_PATH)
    user_content = template.format(
        spatial_context=spatial_context,
        motion_context=motion_context,
        depth_context=depth_context,
        video_description=video_description,
    )

    system, _, user = user_content.partition("\n\nUser:")
    system = system.replace("System:", "").strip()
    user = user.strip() if user else user_content

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
