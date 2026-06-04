"""
PyTorch model architecture for Tribes policy and value prediction.

The model takes a game state (JSON payload from Java bridge) and outputs:
1. Factorized action logits: [action_type_logits, source_logits, target_logits, param_logits]
2. State value estimate for training

State Representation:
- Board: 11x11 grid encoding terrain, resources, buildings, unit presence
- Units: Feature vectors for all units exposed by the Java payload
- Cities: Feature vectors for all cities exposed by the Java payload
- Tech tree: Per-tribe researched technologies
- Tribe stats: Per-tribe stars, score, production, and combat/economy counters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
from typing import Dict, Tuple, Optional
from pathlib import Path


DEFAULT_ACTION_SIZES = {
    "action_type": 32,
    "source_actor": 151,
    "target_actor": 284,
    "param": 80,
}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_action_space_sizes(schema_path: Optional[Path] = None) -> Dict[str, int]:
    """Load action head sizes from the action space schema."""
    if schema_path is None:
        schema_path = Path(__file__).parent / "action_space_schema.json"
    if not schema_path.exists():
        return DEFAULT_ACTION_SIZES.copy()

    try:
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        components = schema.get("components", {})
        return {
            "action_type": int(components.get("action_type", {}).get("size", DEFAULT_ACTION_SIZES["action_type"])),
            "source_actor": int(components.get("source_actor", {}).get("size", DEFAULT_ACTION_SIZES["source_actor"])),
            "target_actor": int(components.get("target_actor", {}).get("size", DEFAULT_ACTION_SIZES["target_actor"])),
            "param": int(components.get("param", {}).get("size", DEFAULT_ACTION_SIZES["param"])),
        }
    except Exception:
        return DEFAULT_ACTION_SIZES.copy()


def load_action_type_index(action_type: str, default: int) -> int:
    schema_path = Path(__file__).parent / "action_space_schema.json"
    if not schema_path.exists():
        return default

    try:
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        index_map = schema.get("components", {}).get("action_type", {}).get("index_map", {})
        return int(index_map.get(action_type, default))
    except Exception:
        return default


class StateEncoder:
    """Encodes game state JSON payload to tensors suitable for NN input."""
    
    def __init__(
        self,
        board_size: int = 11,
        max_units: int = 100,
        max_cities: int = 50,
        max_tribes: int = 12,
    ):
        self.board_size = board_size
        self.max_units = max_units
        self.max_cities = max_cities
        self.max_tribes = max_tribes
        
        # Feature dimensions (will be flattened into state vector)
        # Board: 11x11 with terrain plus occupancy channels from the provided payload.
        # For simplicity, we'll flatten the board and unit/city features
        self.board_channels = 16
        self.board_features = board_size * board_size * self.board_channels
        self.unit_features = max_units * 16
        self.city_features = max_cities * 10
        self.tech_features_per_tribe = 50
        self.tech_features = max_tribes * self.tech_features_per_tribe
        self.tribe_features_per_tribe = 12
        self.tribe_features = max_tribes * self.tribe_features_per_tribe
        
        self.total_state_size = (
            self.board_features + 
            self.unit_features + 
            self.city_features + 
            self.tech_features + 
            self.tribe_features
        )
    
    def encode_board(self, board_dict: Dict, active_tribe_id: int, tribes_list: list) -> np.ndarray:
        """
        Encode board into tensor.
        
        Each tile gets 16 channels:
        - Terrain including UNKNOWN for hidden tiles
        - Visibility
        - Resource/building presence on visible tiles
        - Visible unit/city ownership relative to the active tribe
        """
        size = board_dict.get("size", self.board_size)
        board_tensor = np.zeros((size, size, self.board_channels), dtype=np.float32)
        
        terrain_map = {
            "UNKNOWN": 0, "WATER": 1, "SHALLOW_WATER": 2, "GRASS": 3,
            "FOREST": 4, "MOUNTAIN": 5, "SNOW": 6, "DESERT": 7, "CITY": 8
        }

        city_owner = {}
        for tribe_dict in tribes_list:
            tribe_id = tribe_dict.get("tribe_id")
            for city in tribe_dict.get("cities", []):
                city_id = city.get("actor_id")
                if city_id is not None:
                    city_owner[int(city_id)] = tribe_id
        
        tiles = board_dict.get("tiles", [])
        for x_row in tiles:
            for y_tile in x_row:
                x, y = y_tile.get("x"), y_tile.get("y")
                if 0 <= x < size and 0 <= y < size:
                    visible = bool(y_tile.get("visible", True))
                    terrain_name = y_tile.get("terrain", "UNKNOWN" if not visible else "GRASS")
                    terrain_idx = terrain_map.get(terrain_name, terrain_map["GRASS"])
                    board_tensor[x, y, terrain_idx] = 1.0
                    board_tensor[x, y, 9] = 1.0 if visible else 0.0

                    if not visible:
                        continue
                    
                    board_tensor[x, y, 10] = 1.0 if y_tile.get("resource") is not None else 0.0
                    board_tensor[x, y, 11] = 1.0 if y_tile.get("building") is not None else 0.0

                    unit = y_tile.get("unit")
                    if unit:
                        if unit.get("tribe_id") == active_tribe_id:
                            board_tensor[x, y, 12] = 1.0
                        else:
                            board_tensor[x, y, 13] = 1.0

                    city_id = y_tile.get("city_id", -1)
                    owner = city_owner.get(city_id)
                    if owner is not None:
                        if owner == active_tribe_id:
                            board_tensor[x, y, 14] = 1.0
                        else:
                            board_tensor[x, y, 15] = 1.0
        
        return board_tensor.flatten()
    
    def encode_units(self, active_tribe_id: int, tribes_list: list) -> np.ndarray:
        """Encode visible unit features, with ownership relative to the active tribe."""
        unit_tensor = np.zeros(self.max_units * 16, dtype=np.float32)
        
        unit_type_map = {
            "WARRIOR": 0, "ARCHER": 1, "KNIGHT": 2, "DEFENDER": 3,
            "SWORDSMAN": 4, "RIDER": 5, "CATAPULT": 6, "BOAT": 7,
            "SHIP": 8, "BATTLESHIP": 9, "MIND_BENDER": 10
        }
        
        unit_idx = 0
        for tribe_dict in tribes_list:
            tribe_id = tribe_dict.get("tribe_id")
            
            for unit in tribe_dict.get("units", []):
                if unit_idx >= self.max_units:
                    break
                
                # Extract features (16 per unit)
                unit_type = unit_type_map.get(unit.get("type", "WARRIOR"), 0)
                health_pct = unit.get("current_hp", 1) / max(unit.get("max_hp", 1), 1)
                position = [unit.get("x", 0) / self.board_size, unit.get("y", 0) / self.board_size]
                atk_val = unit.get("atk", 1) / 5.0
                def_val = unit.get("def", 1) / 5.0
                mov_val = unit.get("mov", 1) / 3.0
                
                offset = unit_idx * 16
                unit_tensor[offset + 0] = float(unit_type) / 10.0
                unit_tensor[offset + 1] = health_pct
                unit_tensor[offset + 2] = position[0]
                unit_tensor[offset + 3] = position[1]
                unit_tensor[offset + 4] = atk_val
                unit_tensor[offset + 5] = def_val
                unit_tensor[offset + 6] = mov_val
                unit_tensor[offset + 7] = 1.0 if unit.get("has_moved") else 0.0
                unit_tensor[offset + 8] = 1.0 if unit.get("has_attacked") else 0.0
                unit_tensor[offset + 9] = 1.0 if unit.get("is_veteran") else 0.0
                unit_tensor[offset + 10] = 1.0 if tribe_id == active_tribe_id else 0.0
                unit_tensor[offset + 11] = 1.0 if tribe_id != active_tribe_id else 0.0
                unit_tensor[offset + 12] = float(tribe_id or 0) / 12.0
                unit_tensor[offset + 13] = float(unit.get("city_id", 0) <= 0)
                unit_tensor[offset + 14] = 1.0
                unit_tensor[offset + 15] = 0.0
                
                unit_idx += 1
        
        return unit_tensor
    
    def encode_cities(self, active_tribe_id: int, tribes_list: list) -> np.ndarray:
        """Encode visible city features, with ownership relative to the active tribe."""
        city_tensor = np.zeros(self.max_cities * 10, dtype=np.float32)
        
        city_idx = 0
        for tribe_dict in tribes_list:
            tribe_id = tribe_dict.get("tribe_id")
            
            for city in tribe_dict.get("cities", []):
                if city_idx >= self.max_cities:
                    break
                
                level_norm = city.get("level", 1) / 5.0
                population_norm = city.get("population", 1) / 50.0
                position = [city.get("x", 0) / self.board_size, city.get("y", 0) / self.board_size]
                is_capital = 1.0 if city.get("is_capital") else 0.0
                has_walls = 1.0 if city.get("has_walls") else 0.0
                
                offset = city_idx * 10
                city_tensor[offset + 0] = level_norm
                city_tensor[offset + 1] = population_norm
                city_tensor[offset + 2] = position[0]
                city_tensor[offset + 3] = position[1]
                city_tensor[offset + 4] = is_capital
                city_tensor[offset + 5] = has_walls
                city_tensor[offset + 6] = 1.0 if tribe_id == active_tribe_id else 0.0
                city_tensor[offset + 7] = 1.0 if tribe_id != active_tribe_id else 0.0
                city_tensor[offset + 8] = float(tribe_id or 0) / 12.0
                city_tensor[offset + 9] = 1.0
                
                city_idx += 1
        
        return city_tensor
    
    def encode_tech_and_tribe(self, active_tribe_id: int, tribes_list: list) -> np.ndarray:
        """Encode per-tribe technology and economy/military stats."""
        features = np.zeros(self.tech_features + self.tribe_features, dtype=np.float32)
        try:
            active_tribe_idx = int(active_tribe_id)
        except (TypeError, ValueError):
            active_tribe_idx = -1
        
        for tribe_dict in tribes_list:
            tribe_id = tribe_dict.get("tribe_id")
            if tribe_id is None:
                continue

            try:
                tribe_idx = int(tribe_id)
            except (TypeError, ValueError):
                continue

            if tribe_idx < 0 or tribe_idx >= self.max_tribes:
                continue

            # Tech tree for every exposed tribe. Hidden opponent tech remains all-zero
            # when Java is configured for partial observability.
            tech_data = tribe_dict.get("technology", {})
            researched_flags = []
            if isinstance(tech_data, dict):
                researched_flags = tech_data.get("researched_flags", tech_data.get("researched_techs", []))
            if researched_flags:
                tech_offset = tribe_idx * self.tech_features_per_tribe
                for i, tech_researched in enumerate(researched_flags[:self.tech_features_per_tribe]):
                    features[tech_offset + i] = 1.0 if tech_researched else 0.0

            cities = tribe_dict.get("cities", [])
            units = tribe_dict.get("units", [])
            production = sum((city.get("production", 0) or 0) for city in cities if isinstance(city, dict))

            offset = self.tech_features + tribe_idx * self.tribe_features_per_tribe
            features[offset + 0] = 1.0
            features[offset + 1] = 1.0 if tribe_idx == active_tribe_idx else 0.0
            features[offset + 2] = min(tribe_dict.get("stars", 0) / 100.0, 1.0)
            features[offset + 3] = min(tribe_dict.get("score", 0) / 1000.0, 1.0)
            features[offset + 4] = min(tribe_dict.get("stars_sent", 0) / 30.0, 1.0)
            features[offset + 5] = min(tribe_dict.get("n_kills", 0) / 20.0, 1.0)
            features[offset + 6] = min(tribe_dict.get("n_wars_declared", 0) / 10.0, 1.0)
            features[offset + 7] = min(tribe_dict.get("n_stars_sent", 0) / 20.0, 1.0)
            features[offset + 8] = min(len(cities) / max(self.max_cities, 1), 1.0)
            features[offset + 9] = min(len(units) / max(self.max_units, 1), 1.0)
            features[offset + 10] = min(production / 50.0, 1.0)
            features[offset + 11] = 1.0 if tribe_dict.get("capital_id", -1) >= 0 else 0.0
        
        return features
    
    def encode(self, payload: Dict) -> torch.Tensor:
        """
        Encode the game state payload to a single state tensor.

        This encoder intentionally trusts the Java payload's observability mode.
        Full-observation training should send full board, unit, city, tech, and
        tribe-stat data; partial-observation runs should send already-redacted data.
        
        Args:
            payload: JSON payload from Java PythonBridge
        
        Returns:
            State tensor of shape (state_size,)
        """
        active_tribe_id = payload.get("active_tribe_id")
        tribes = payload.get("tribes", [])
        board_features = self.encode_board(payload.get("board", {}), active_tribe_id, tribes)
        unit_features = self.encode_units(active_tribe_id, tribes)
        city_features = self.encode_cities(active_tribe_id, tribes)
        tech_tribe_features = self.encode_tech_and_tribe(active_tribe_id, tribes)
        
        state = np.concatenate([
            board_features,
            unit_features,
            city_features,
            tech_tribe_features
        ])
        
        return torch.from_numpy(state).float()


class TribesModel(nn.Module):
    """
    Policy and value head model for Tribes game.
    
    Outputs:
    - action_type_logits: (batch_size, action_type_size)
    - source_logits: (batch_size, source_actor_size)
    - target_logits: (batch_size, target_actor_size)
    - param_logits: (batch_size, param_size)
    - value: (batch_size, 1)
    """
    
    def __init__(self, state_size: int = None, mask_send_stars: bool = True):
        super().__init__()
        
        # If state_size not provided, use encoder's calculated size
        if state_size is None:
            encoder = StateEncoder()
            state_size = encoder.total_state_size
        
        self.state_size = state_size
        self.hidden_size = 512
        
        action_sizes = load_action_space_sizes()
        self.action_type_size = action_sizes["action_type"]
        self.source_actor_size = action_sizes["source_actor"]
        self.target_actor_size = action_sizes["target_actor"]
        self.param_size = action_sizes["param"]
        self.mask_send_stars = mask_send_stars
        self.send_stars_action_type_index = load_action_type_index("SEND_STARS", 14)

        # Shared trunk: process state to hidden representation
        self.trunk = nn.Sequential(
            nn.Linear(state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        
        # Policy heads: one for each action component
        self.action_type_head = nn.Linear(self.hidden_size, self.action_type_size)
        self.source_head = nn.Linear(self.hidden_size, self.source_actor_size)
        self.target_head = nn.Linear(self.hidden_size, self.target_actor_size)
        self.param_head = nn.Linear(self.hidden_size, self.param_size)
        
        # Value head for training
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: (batch_size, state_size) tensor
        
        Returns:
            Tuple of (action_type_logits, source_logits, target_logits, param_logits, value)
        """
        hidden = self.trunk(state)
        
        action_type_logits = self.action_type_head(hidden)
        if self.mask_send_stars and 0 <= self.send_stars_action_type_index < action_type_logits.shape[-1]:
            action_type_logits = action_type_logits.clone()
            action_type_logits[:, self.send_stars_action_type_index] = float("-inf")
        source_logits = self.source_head(hidden)
        target_logits = self.target_head(hidden)
        param_logits = self.param_head(hidden)
        value = self.value_head(hidden)
        
        return action_type_logits, source_logits, target_logits, param_logits, value
    
    def save(self, path: str) -> None:
        """Save model weights to disk."""
        output_path = Path(path)
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(self.state_dict(), temp_path)
        temp_path.replace(output_path)
    
    def load(self, path: str, device: str = "cpu") -> None:
        """Load model weights from disk."""
        self.load_state_dict(torch.load(path, map_location=device))
    
    @staticmethod
    def create_or_load(model_path: Optional[str] = None, device: str = "cpu") -> "TribesModel":
        """Factory method to create or load model."""
        model = TribesModel(mask_send_stars=env_bool("TRIBES_MASK_SEND_STARS", True))
        if model_path and Path(model_path).exists():
            model.load(model_path, device=device)
        return model.to(device)

class TribesTransformerModel(nn.Module):
    """
    Minimal but correct Transformer policy/value model.
    Keeps identical input/output interface.
    """

    def __init__(
        self,
        state_size: int = None,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        mask_send_stars: bool = True,
    ):
        super().__init__()

        encoder_layout = StateEncoder()
        if state_size is None:
            state_size = encoder_layout.total_state_size

        self.state_size = state_size
        self.d_model = d_model

        action_sizes = load_action_space_sizes()
        self.action_type_size = action_sizes["action_type"]
        self.source_size = action_sizes["source_actor"]
        self.target_size = action_sizes["target_actor"]
        self.param_size = action_sizes["param"]
        self.mask_send_stars = mask_send_stars
        self.send_stars_action_type_index = load_action_type_index("SEND_STARS", 14)

        # -------------------------
        # TOKEN SIZES (fixed layout from encoder)
        # -------------------------
        self.num_tiles = encoder_layout.board_size * encoder_layout.board_size
        self.tile_dim = encoder_layout.board_channels

        self.num_units = encoder_layout.max_units
        self.unit_dim = 16

        self.num_cities = encoder_layout.max_cities
        self.city_dim = 10

        self.tech_dim = encoder_layout.tech_features
        self.tribe_dim = encoder_layout.tribe_features

        # -------------------------
        # EMBEDDINGS
        # -------------------------
        self.tile_embed = nn.Linear(self.tile_dim, d_model)
        self.unit_embed = nn.Linear(self.unit_dim, d_model)
        self.city_embed = nn.Linear(self.city_dim, d_model)
        self.tech_embed = nn.Linear(self.tech_dim, d_model)
        self.tribe_embed = nn.Linear(self.tribe_dim, d_model)

        # learned index embeddings (better than 2D grid pos encoding)
        self.tile_index_embed = nn.Embedding(self.num_tiles, d_model)
        self.unit_index_embed = nn.Embedding(self.num_units, d_model)
        self.city_index_embed = nn.Embedding(self.num_cities, d_model)

        # global token
        self.global_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # -------------------------
        # TRANSFORMER BACKBONE
        # -------------------------
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # -------------------------
        # MOVE HEADS (CROSS ATTENTION STYLE)
        # -------------------------
        self.q_tile = nn.Linear(d_model, d_model)
        self.k_tile = nn.Linear(d_model, d_model)

        self.q_unit = nn.Linear(d_model, d_model)
        self.k_unit = nn.Linear(d_model, d_model)

        # action heads
        self.action_type_head = nn.Linear(d_model, self.action_type_size)
        self.source_head = nn.Linear(d_model, self.source_size)
        self.target_head = nn.Linear(d_model, self.target_size)
        self.param_head = nn.Linear(d_model, self.param_size)

        # value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, masks=None):
        """
        state: (B, state_size)
        masks: optional dict of action masks (for logits masking)
        """

        B = state.size(0)
        offset = 0

        # -------------------------
        # TILE TOKENS
        # -------------------------
        tiles = state[:, offset:offset + self.num_tiles * self.tile_dim]
        tiles = tiles.view(B, self.num_tiles, self.tile_dim)
        offset += self.num_tiles * self.tile_dim

        tile_idx = torch.arange(self.num_tiles, device=state.device)
        tile_idx = tile_idx.unsqueeze(0).expand(B, -1)

        tile_tokens = self.tile_embed(tiles) + self.tile_index_embed(tile_idx)

        # -------------------------
        # UNIT TOKENS
        # -------------------------
        units = state[:, offset:offset + self.num_units * self.unit_dim]
        units = units.view(B, self.num_units, self.unit_dim)
        offset += self.num_units * self.unit_dim

        unit_idx = torch.arange(self.num_units, device=state.device)
        unit_idx = unit_idx.unsqueeze(0).expand(B, -1)

        unit_tokens = self.unit_embed(units) + self.unit_index_embed(unit_idx)

        # -------------------------
        # CITY TOKENS
        # -------------------------
        cities = state[:, offset:offset + self.num_cities * self.city_dim]
        cities = cities.view(B, self.num_cities, self.city_dim)
        offset += self.num_cities * self.city_dim

        city_idx = torch.arange(self.num_cities, device=state.device)
        city_idx = city_idx.unsqueeze(0).expand(B, -1)

        city_tokens = self.city_embed(cities) + self.city_index_embed(city_idx)

        # -------------------------
        # GLOBAL FEATURES
        # -------------------------
        tech = state[:, offset:offset + self.tech_dim]
        offset += self.tech_dim

        tribe = state[:, offset:offset + self.tribe_dim]

        tech_tok = self.tech_embed(tech).unsqueeze(1)
        tribe_tok = self.tribe_embed(tribe).unsqueeze(1)

        # -------------------------
        # CONCAT TOKENS
        # -------------------------
        global_tok = self.global_token.expand(B, 1, self.d_model)

        tokens = torch.cat([
            global_tok,
            tile_tokens,
            unit_tokens,
            city_tokens,
            tech_tok,
            tribe_tok
        ], dim=1)

        # -------------------------
        # TRANSFORMER ENCODING
        # -------------------------
        tokens = self.encoder(tokens)

        global_vec = tokens[:, 0]

        # =========================================================
        # POLICY HEADS
        # =========================================================

        action_type_logits = self.action_type_head(global_vec)
        if self.mask_send_stars and 0 <= self.send_stars_action_type_index < action_type_logits.shape[-1]:
            action_type_logits = action_type_logits.clone()
            action_type_logits[:, self.send_stars_action_type_index] = float("-inf")
        source_logits = self.source_head(global_vec)
        target_logits = self.target_head(global_vec)
        param_logits = self.param_head(global_vec)

        # =========================================================
        # CROSS-ATTENTION MOVE HEAD (tile-to-tile)
        # =========================================================
        tile_repr = tokens[:, 1:1 + self.num_tiles]

        q = self.q_tile(tile_repr)          # (B, T, D)
        k = self.k_tile(tile_repr)          # (B, T, D)

        move_logits = torch.einsum("btd,bsd->bts", q, k) / (self.d_model ** 0.5)

        # you can later reshape this into your source/target factorization
        # or inject into source/target heads if you want

        # =========================================================
        # VALUE HEAD
        # =========================================================
        value = self.value_head(global_vec)

        return action_type_logits, source_logits, target_logits, param_logits, value
    def save(self, path: str) -> None:
        """Save model weights to disk."""
        output_path = Path(path)
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(self.state_dict(), temp_path)
        temp_path.replace(output_path)

    def load(self, path: str, device: str = "cpu") -> None:
        """Load model weights from disk."""
        self.load_state_dict(torch.load(path, map_location=device))

    @staticmethod
    def create_or_load(
        model_path: Optional[str] = None,
        device: str = "cpu"
    ) -> "TribesTransformerModel":
        """Factory method to create or load model."""
        model = TribesTransformerModel(mask_send_stars=env_bool("TRIBES_MASK_SEND_STARS", True))

        if model_path and Path(model_path).exists():
            model.load(model_path, device=device)

        return model.to(device)


def encode_state(payload: Dict, encoder: Optional[StateEncoder] = None) -> torch.Tensor:
    """Convenience function to encode a state payload."""
    if encoder is None:
        encoder = StateEncoder()
    return encoder.encode(payload)


def load_model(model_path: str, device: str = "cpu") -> TribesModel:
    """Load a trained model from disk."""
    return TribesModel.create_or_load(model_path=model_path, device=device)


# Test/example usage
if __name__ == "__main__":
    # Get the actual state size from the encoder
    encoder = StateEncoder()
    state_size = encoder.total_state_size
    
    # Example: Create model and do a forward pass
    model = TribesTransformerModel(state_size=encoder.total_state_size)
    model.eval()
    
    # Dummy state tensor (batch_size=2)
    dummy_state = torch.randn(2, state_size)
    
    with torch.no_grad():
        action_type_logits, source_logits, target_logits, param_logits, value = model(dummy_state)
    
    print(f"action_type_logits shape: {action_type_logits.shape}")  # (2, 32)
    print(f"source_logits shape: {source_logits.shape}")            # (2, 151)
    print(f"target_logits shape: {target_logits.shape}")            # (2, 284)
    print(f"param_logits shape: {param_logits.shape}")              # (2, 80)
    print(f"value shape: {value.shape}")                            # (2, 1)
    
    # Test StateEncoder with JSON
    print("\n--- Testing StateEncoder ---")
    encoder = StateEncoder()
    print(f"Total state size: {encoder.total_state_size}")
    
    # Try loading examplePayload
    with open("examplePayload.json", "r") as f:
        payload = json.load(f)
    
    state_tensor = encoder.encode(payload)
    print(f"Encoded state shape: {state_tensor.shape}")
    print(f"Expected shape: ({encoder.total_state_size},)")
    
    # Forward pass with real state
    with torch.no_grad():
        batch_state = state_tensor.unsqueeze(0)  # Add batch dimension
        out = model(batch_state)
        print(f"\nForward pass with real state:")
        print(f"  action_type_logits: {out[0].shape}")
        print(f"  source_logits: {out[1].shape}")
        print(f"  target_logits: {out[2].shape}")
        print(f"  param_logits: {out[3].shape}")
        print(f"  value: {out[4].shape}")
