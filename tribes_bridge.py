from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


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


class TribesBridge:
	def __init__(self, compile_first: bool = True) -> None:
		if compile_first:
			compile_java()

		classpath = f"{BUILD_DIR}{os.pathsep}{LIB_DIR / 'json.jar'}"
		self._process = subprocess.Popen(
			["java", "-cp", classpath, "core.game.BridgeServer"],
			cwd=TRIBES_DIR,
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			bufsize=1,
		)
		self._stderr_queue: queue.Queue[str] = queue.Queue()
		self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
		self._stderr_thread.start()

	def _resolve_path(self, path: str) -> str:
		candidate = Path(path)
		if candidate.is_absolute():
			return str(candidate)
		root_candidate = ROOT / candidate
		if root_candidate.exists():
			return str(root_candidate)
		tribes_candidate = TRIBES_DIR / candidate
		if tribes_candidate.exists():
			return str(tribes_candidate)
		return str(root_candidate)

	def _drain_stderr(self) -> None:
		assert self._process.stderr is not None
		for line in self._process.stderr:
			self._stderr_queue.put(line)
			sys.stderr.write(line)

	def _request(self, message: dict[str, Any]) -> dict[str, Any]:
		if self._process.poll() is not None:
			raise RuntimeError("Bridge process has exited")
		assert self._process.stdin is not None
		assert self._process.stdout is not None
		self._process.stdin.write(json.dumps(message) + "\n")
		self._process.stdin.flush()
		response_line = self._process.stdout.readline()
		if not response_line:
			raise RuntimeError("Bridge process closed the pipe")
		response = json.loads(response_line)
		if not response.get("ok", False):
			raise RuntimeError(response.get("error", "Unknown bridge error"))
		return response

	def reset(
		self,
		*,
		level_file: str | None = None,
		level_seed: int | None = None,
		tribes: list[int | str] | None = None,
		game_mode: str | int = "SCORE",
		seed: int | None = None,
	) -> dict[str, Any]:
		message: dict[str, Any] = {"cmd": "reset", "gameMode": game_mode}
		if seed is not None:
			message["seed"] = seed
		if level_file is not None:
			message["levelFile"] = self._resolve_path(level_file)
		elif level_seed is not None and tribes is not None:
			message["levelSeed"] = level_seed
			message["tribes"] = tribes
		else:
			raise ValueError("reset requires level_file or level_seed + tribes")
		return self._request(message)["state"]

	def observe(self) -> dict[str, Any]:
		return self._request({"cmd": "observe"})["state"]

	def actions(self) -> list[dict[str, Any]]:
		return self._request({"cmd": "actions"})["actions"]

	def step(self, action_index: int) -> dict[str, Any]:
		return self._request({"cmd": "step", "actionIndex": action_index})

	def close(self) -> None:
		if self._process.poll() is None:
			try:
				self._request({"cmd": "close"})
			except Exception:
				pass
			self._process.terminate()

	def __enter__(self) -> "TribesBridge":
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		self.close()