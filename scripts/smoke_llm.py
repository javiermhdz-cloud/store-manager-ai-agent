"""Send a single manual smoke-test prompt through the configured model."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm import generate


def main() -> None:
    response, usage = generate("Hola", interaction_type="smoke")
    print(response.text)
    print(usage)


if __name__ == "__main__":
    main()