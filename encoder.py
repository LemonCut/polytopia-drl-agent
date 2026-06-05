from __future__ import annotations

import json
import math
import zlib
from typing import Any

import numpy as np

ENGINEERED_FEATURE_DIM = 1024
RAW_FEATURE_DIM = 256
DEFAULT_ENCODER_MODE = "combined"


class JsonStateEncoder:
    def __init__(self, mode: str = DEFAULT_ENCODER_MODE, engineered_dim: int = ENGINEERED_FEATURE_DIM, raw_dim: int = RAW_FEATURE_DIM) -> None:
        self.mode = mode
        self.engineered_dim = int(engineered_dim)
        self.raw_dim = int(raw_dim)

    @property
    def output_dim(self) -> int:
        if self.mode == "engineered":
            return self.engineered_dim
        if self.mode == "raw":
            return self.raw_dim
        if self.mode == "combined":
            return self.engineered_dim + self.raw_dim
        raise ValueError(f"Unknown encoder mode: {self.mode}")

    def encode(self, state_json: str) -> np.ndarray:
        if self.mode == "engineered":
            return self._encode_engineered(state_json)
        if self.mode == "raw":
            return self._encode_raw(state_json)
        if self.mode == "combined":
            return np.concatenate([self._encode_engineered(state_json), self._encode_raw(state_json)], axis=0)
        raise ValueError(f"Unknown encoder mode: {self.mode}")

    def _encode_engineered(self, state_json: str) -> np.ndarray:
        vector = np.zeros(self.engineered_dim, dtype=np.float32)
        try:
            state = json.loads(state_json)
        except json.JSONDecodeError:
            return vector

        features: list[float] = []

        def add(value: float) -> None:
            if len(features) < self.engineered_dim:
                features.append(float(value))

        def scaled(value: float, scale: float = 10.0) -> float:
            return math.tanh(float(value) / float(scale))

        def safe_number(value: Any, default: float = 0.0) -> float:
            try:
                if isinstance(value, bool):
                    return 1.0 if value else 0.0
                return float(value)
            except Exception:
                return default

        def encode_string(value: Any) -> None:
            if not isinstance(value, str):
                add(0.0)
                return
            add(float(len(value)) / 64.0)

        def encode_stats(values: Iterable[float], *, scale: float = 10.0) -> None:
            data = [float(value) for value in values]
            if not data:
                add(0.0)
                add(0.0)
                add(0.0)
                add(0.0)
                return
            array = np.asarray(data, dtype=np.float32)
            add(float(array.mean()) / scale)
            add(float(array.std()) / scale)
            add(float(array.min()) / scale)
            add(float(array.max()) / scale)

        def encode_grid(grid: Any, histogram_max: int = 8) -> None:
            if not isinstance(grid, list) or not grid:
                for _ in range(6 + histogram_max):
                    add(0.0)
                return
            flattened: list[float] = []
            for row in grid:
                if isinstance(row, list):
                    flattened.extend(float(value) for value in row if isinstance(value, (int, float, np.integer, np.floating, bool)))
                elif isinstance(row, (int, float, np.integer, np.floating, bool)):
                    flattened.append(float(row))
            if not flattened:
                add(0.0)
                add(0.0)
                add(0.0)
                add(0.0)
                add(0.0)
                add(0.0)
                return
            array = np.asarray(flattened, dtype=np.float32)
            add(float((array != 0).mean()))
            add(float(array.mean()) / 10.0)
            add(float(array.std()) / 10.0)
            add(float(array.min()) / 10.0)
            add(float(array.max()) / 10.0)
            add(float(len(np.unique(array))) / 32.0)
            for value in range(histogram_max):
                add(float(np.sum(array == value)) / max(1.0, float(array.size)))

        def encode_board(board: dict[str, Any]) -> None:
            encode_grid(board.get("terrain"))
            encode_grid(board.get("resource"))
            encode_grid(board.get("building"))
            encode_grid(board.get("unitID"))
            encode_grid(board.get("cityID"))
            encode_grid(board.get("network"), histogram_max=2)
            add(safe_number(board.get("actorIDcounter")) / 32.0)

        def encode_active_tribe(tribe: dict[str, Any]) -> None:
            add(safe_number(tribe.get("score")) / 100.0)
            add(safe_number(tribe.get("star")) / 20.0)
            add(safe_number(tribe.get("nKills")) / 20.0)
            add(safe_number(tribe.get("nPacifistCount")) / 20.0)
            add(safe_number(len(tribe.get("citiesID", []))) / 20.0)
            add(safe_number(len(tribe.get("extraUnits", []))) / 20.0)
            add(safe_number(len(tribe.get("monuments", []))) / 20.0)
            add(safe_number(len(tribe.get("tribesMet", []))) / 20.0)
            add(1.0 if tribe.get("winner") == 0 else 0.0)
            add(1.0 if tribe.get("winner") == 1 else 0.0)
            add(1.0 if tribe.get("winner") == -1 else 0.0)
            technology = tribe.get("technology", {})
            researched = technology.get("researched", []) if isinstance(technology, dict) else []
            add(float(sum(bool(item) for item in researched)) / max(1.0, float(len(researched))))
            add(1.0 if isinstance(technology, dict) and technology.get("everythingResearched") else 0.0)
            encode_string(tribe.get("type"))

        def encode_units(units: dict[str, Any]) -> None:
            if not isinstance(units, dict) or not units:
                for _ in range(29):
                    add(0.0)
                return
            current_hp = []
            veteran = []
            kills = []
            xs = []
            ys = []
            city_ids = []
            tribe_ids = []
            for unit in units.values():
                if not isinstance(unit, dict):
                    continue
                current_hp.append(safe_number(unit.get("currentHP")))
                veteran.append(1.0 if unit.get("isVeteran") else 0.0)
                kills.append(safe_number(unit.get("kill")))
                xs.append(safe_number(unit.get("x")))
                ys.append(safe_number(unit.get("y")))
                city_ids.append(safe_number(unit.get("cityID")))
                tribe_ids.append(safe_number(unit.get("tribeId")))
            add(float(len(units)) / 20.0)
            encode_stats(current_hp, scale=10.0)
            encode_stats(veteran, scale=1.0)
            encode_stats(kills, scale=10.0)
            encode_stats(xs, scale=16.0)
            encode_stats(ys, scale=16.0)
            encode_stats(city_ids, scale=10.0)
            encode_stats(tribe_ids, scale=10.0)

        def encode_cities(cities: dict[str, Any]) -> None:
            if not isinstance(cities, dict) or not cities:
                for _ in range(33):
                    add(0.0)
                return
            level = []
            population = []
            production = []
            walls = []
            capital = []
            xs = []
            ys = []
            units = []
            for city in cities.values():
                if not isinstance(city, dict):
                    continue
                level.append(safe_number(city.get("level")))
                population.append(safe_number(city.get("population")))
                production.append(safe_number(city.get("production")))
                walls.append(1.0 if city.get("hasWalls") else 0.0)
                capital.append(1.0 if city.get("isCapital") else 0.0)
                xs.append(safe_number(city.get("x")))
                ys.append(safe_number(city.get("y")))
                units.append(safe_number(len(city.get("units", []))))
            add(float(len(cities)) / 20.0)
            encode_stats(level, scale=10.0)
            encode_stats(population, scale=10.0)
            encode_stats(production, scale=10.0)
            encode_stats(walls, scale=1.0)
            encode_stats(capital, scale=1.0)
            encode_stats(xs, scale=16.0)
            encode_stats(ys, scale=16.0)
            encode_stats(units, scale=10.0)

        def encode_all_tribes(all_tribes: dict[str, Any], active_tribe_id: str) -> None:
            if not isinstance(all_tribes, dict) or not all_tribes:
                for _ in range(24):
                    add(0.0)
                return
            scores: list[float] = []
            stars: list[float] = []
            city_counts: list[float] = []
            unit_counts: list[float] = []
            tech_completion: list[float] = []
            active_score = 0.0
            active_stars = 0.0
            for tribe_id, tribe in all_tribes.items():
                if not isinstance(tribe, dict):
                    continue
                score = safe_number(tribe.get("score"))
                star = safe_number(tribe.get("star"))
                city_count = float(len(tribe.get("citiesID", [])))
                unit_count = float(len(tribe.get("extraUnits", [])))
                technology = tribe.get("technology", {})
                researched = technology.get("researched", []) if isinstance(technology, dict) else []
                completion = float(sum(bool(item) for item in researched)) / max(1.0, float(len(researched)))
                scores.append(score)
                stars.append(star)
                city_counts.append(city_count)
                unit_counts.append(unit_count)
                tech_completion.append(completion)
                if str(tribe_id) == active_tribe_id:
                    active_score = score
                    active_stars = star
            add(float(len(scores)) / 8.0)
            encode_stats(scores, scale=100.0)
            encode_stats(stars, scale=20.0)
            encode_stats(city_counts, scale=20.0)
            encode_stats(unit_counts, scale=20.0)
            encode_stats(tech_completion, scale=1.0)
            best_score = max(scores)
            mean_score = float(np.mean(scores))
            add((active_score - mean_score) / 100.0)
            add((active_score - best_score) / 100.0)
            add(active_stars / 20.0)
            add(1.0 if active_score >= best_score else 0.0)

        board = state.get("board", {})
        tribes = state.get("tribes", {})
        active_tribe_id = str(state.get("activeTribeID"))
        active_tribe = tribes.get(active_tribe_id, {}) if isinstance(tribes, dict) else {}

        add(safe_number(state.get("tick")) / 100.0)
        add(1.0 if state.get("gameIsOver") else 0.0)
        encode_string(state.get("gameMode"))
        encode_board(board if isinstance(board, dict) else {})
        encode_all_tribes(tribes if isinstance(tribes, dict) else {}, active_tribe_id)
        encode_active_tribe(active_tribe if isinstance(active_tribe, dict) else {})
        encode_units(state.get("unit", {}))
        encode_cities(state.get("city", {}))

        numeric = np.asarray(features, dtype=np.float32)
        vector[: len(numeric)] = numeric
        if len(numeric) < self.engineered_dim:
            vector[len(numeric) :] = 0.0
        return vector

    def _encode_raw(self, state_json: str) -> np.ndarray:
        vector = np.zeros(self.raw_dim, dtype=np.float32)
        if not state_json:
            return vector
        text = state_json.lower()
        window = 3
        if len(text) < window:
            index = zlib.crc32(text.encode("utf-8")) % self.raw_dim
            vector[index] += 1.0
            return vector
        for start in range(len(text) - window + 1):
            ngram = text[start : start + window]
            index = zlib.crc32(ngram.encode("utf-8")) % self.raw_dim
            vector[index] += 1.0
        vector /= max(1.0, float(np.linalg.norm(vector)))
        return vector.astype(np.float32, copy=False)
