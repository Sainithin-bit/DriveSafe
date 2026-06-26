from __future__ import annotations

from pathlib import Path

from drivesafe.llm import load_prompt_template

REPO_ROOT = Path(__file__).resolve().parents[1]
RISK_PROMPT_PATH = REPO_ROOT / "configs" / "risk_prompt.txt"


def build_risk_messages(
    caption: str,
    prompt_path: Path | None = None,
) -> list[dict[str, str]]:
    template = load_prompt_template(prompt_path or RISK_PROMPT_PATH)
    user_content = template.format(caption=caption)

    system, _, user = user_content.partition("\n\nInputs:")
    if not user:
        system, _, user = user_content.partition("\n\nExample")
        user = "Inputs: Caption (Cv): " + caption + "\n\n" + user if user else user_content
    system = system.replace("System:", "").strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
