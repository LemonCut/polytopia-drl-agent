from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRIBES_DIR = ROOT / "tribes"
SRC_DIR = TRIBES_DIR / "src"
LIB_DIR = TRIBES_DIR / "lib"
BUILD_DIR = TRIBES_DIR / "build" / "classes"


def compile_java() -> None:
	BUILD_DIR.mkdir(parents=True, exist_ok=True)

	java_files = [str(path) for path in SRC_DIR.rglob("*.java")]
	if not java_files:
		raise FileNotFoundError(f"No Java source files found under {SRC_DIR}")

	classpath = str(LIB_DIR / "json.jar")
	command = [
		"javac",
		"-d",
		str(BUILD_DIR),
		"-cp",
		classpath,
		*java_files,
	]
	subprocess.run(command, cwd=TRIBES_DIR, check=True)


def run_play(args: list[str] | None = None) -> None:
	compile_java()

	classpath = f"{BUILD_DIR}:{LIB_DIR / 'json.jar'}"
	command = ["java", "-cp", classpath, "Play", *(args or [])]
	subprocess.run(command, cwd=TRIBES_DIR, check=True)


if __name__ == "__main__":
	run_play(sys.argv[1:])
