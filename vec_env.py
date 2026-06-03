from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any, Iterable, List, Tuple

from tribes_env import TribesEnv
from encoder import JsonStateEncoder


def _worker_loop(conn: Connection, env_kwargs: dict[str, Any], encoder_kwargs: dict[str, Any]) -> None:
    env = TribesEnv(**env_kwargs)
    encoder = JsonStateEncoder(**(encoder_kwargs or {}))
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                seed = data
                obs, info = env.reset(seed=seed)
                encoded = encoder.encode(obs.get("state_json"))
                conn.send((encoded, info))
            elif cmd == "step":
                action = data
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                encoded = encoder.encode(next_obs.get("state_json"))
                conn.send((encoded, reward, terminated, truncated, next_info))
            elif cmd == "close":
                env.close()
                conn.close()
                break
            else:
                conn.send(RuntimeError(f"Unknown command: {cmd}"))
    except KeyboardInterrupt:
        try:
            env.close()
        except Exception:
            pass


class VecEnv:
    """A minimal multiprocessing vectorized environment using Pipes.

    Methods are blocking but environment step calls run in parallel across processes.
    """

    def __init__(self, num_workers: int = 4, env_kwargs: dict[str, Any] | None = None, encoder_kwargs: dict[str, Any] | None = None) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self.num_workers = int(num_workers)
        self.env_kwargs = env_kwargs or {}
        self.encoder_kwargs = encoder_kwargs or {}
        self.parent_conns: List[Connection] = []
        self.processes: List[mp.Process] = []

        for _ in range(self.num_workers):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=_worker_loop, args=(child_conn, self.env_kwargs, self.encoder_kwargs), daemon=True)
            p.start()
            child_conn.close()
            self.parent_conns.append(parent_conn)
            self.processes.append(p)

    def reset(self, seeds: Iterable[int | None] | None = None) -> List[Tuple[dict[str, Any], dict[str, Any]]]:
        seeds = list(seeds) if seeds is not None else [None] * self.num_workers
        if len(seeds) != self.num_workers:
            raise ValueError("seeds length must match num_workers")
        for conn, seed in zip(self.parent_conns, seeds):
            conn.send(("reset", seed))
        results = [conn.recv() for conn in self.parent_conns]
        return results

    def reset_worker(self, index: int, seed: int | None = None) -> Tuple[dict[str, Any], dict[str, Any]]:
        if index < 0 or index >= self.num_workers:
            raise IndexError("worker index out of range")
        conn = self.parent_conns[index]
        conn.send(("reset", seed))
        return conn.recv()

    def step(self, actions: Iterable[int]) -> List[Tuple[dict[str, Any], float, bool, bool, dict[str, Any]]]:
        actions = list(actions)
        if len(actions) != self.num_workers:
            raise ValueError("actions length must match num_workers")
        for conn, action in zip(self.parent_conns, actions):
            conn.send(("step", int(action)))
        results = [conn.recv() for conn in self.parent_conns]
        return results

    def step_worker(self, index: int, action: int) -> Tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if index < 0 or index >= self.num_workers:
            raise IndexError("worker index out of range")
        conn = self.parent_conns[index]
        conn.send(("step", int(action)))
        return conn.recv()

    def close(self) -> None:
        for conn in self.parent_conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=1.0)
