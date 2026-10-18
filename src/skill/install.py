"""``graphweave install``: copy the agent skill to where assistants look for it.

Claude Code reads skills from ``~/.claude/skills/<name>/SKILL.md`` (or a
project's ``.claude/skills``), and Codex from ``~/.codex/skills/<name>/``. By
default the skill goes to every assistant whose home folder exists, and to
Claude Code when none does.
"""

from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path

SKILL_NAME = "graphweave"

#: Assistant name to the folder its skills live in, under the user's home.
HOME_TARGETS = {
    "claude": Path(".claude") / "skills",
    "codex": Path(".codex") / "skills",
}


def skill_text() -> str:
    """The packaged ``SKILL.md``."""
    return resources.files("graphrag.skill").joinpath("SKILL.md").read_text(encoding="utf-8")


def default_targets(home: Path) -> list[str]:
    """Assistants that appear to be installed, or Claude Code when none does."""
    found = [name for name, sub in HOME_TARGETS.items() if (home / sub.parts[0]).is_dir()]
    return found or ["claude"]


def install(directories: list[Path]) -> list[Path]:
    """Write the skill into each skills directory; return the files written."""
    text = skill_text()
    written = []
    for directory in directories:
        destination = directory / SKILL_NAME / "SKILL.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        written.append(destination)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="graphweave install",
        description="Install the graphweave skill for Claude Code and Codex.",
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--target", choices=[*HOME_TARGETS, "all"],
                       help="which assistant (default: every one found, else claude)")
    where.add_argument("--project", action="store_true",
                       help="install into ./.claude/skills for this project only")
    where.add_argument("--dir", type=Path, help="install into this skills directory")
    args = parser.parse_args(argv)

    home = Path.home()
    if args.dir is not None:
        directories = [args.dir]
    elif args.project:
        directories = [Path.cwd() / ".claude" / "skills"]
    else:
        names = list(HOME_TARGETS) if args.target == "all" else (
            [args.target] if args.target else default_targets(home))
        directories = [home / HOME_TARGETS[name] for name in names]

    try:
        written = install(directories)
    except OSError as exc:
        print(f"graphweave install: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(f"Installed {path}")
    print('Ask your assistant something like: "who worked on retries in psf/requests?"')
    print('Optional MCP tools: pip install "graphweave[mcp]", then e.g. claude mcp add graphweave -- graphweave-mcp')
    return 0
