"""
Action encoding/decoding utilities for Tribes with mixed-radix factorized action space.

Provides functions to:
1. Load and validate action_space_schema.json
2. Convert between action objects and component indices
3. Compute masking vectors for legal actions
4. Map available_actions from game captures to encoded indices
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


class ActionSpaceEncoder:
    """Encodes and decodes actions using mixed-radix factorized representation."""
    
    def __init__(self, schema_path: str = "action_space_schema.json"):
        """
        Load and initialize the action space schema.
        
        Args:
            schema_path: Path to action_space_schema.json
        """
        schema_file = Path(__file__).parent / schema_path
        with open(schema_file, "r") as f:
            self.schema = json.load(f)
        
        self.board_size = self.schema["board_size"]
        self.max_units = self.schema["max_units"]
        self.max_cities = self.schema["max_cities"]
        self.max_tribes = self.schema["max_tribes"]
        
        # Build reverse index maps for action types
        self.action_type_str_to_idx = self.schema["components"]["action_type"]["index_map"]
        self.action_type_idx_to_str = {v: k for k, v in self.action_type_str_to_idx.items()}
        
        # Component sizes
        self.action_type_size = self.schema["components"]["action_type"]["size"]
        self.source_actor_size = self.schema["components"]["source_actor"]["size"]
        self.target_actor_size = self.schema["components"]["target_actor"]["size"]
        self.param_size = self.schema["components"]["param"]["size"]
        
        # Total flattened action space (used for flat indexing if needed later)
        self.total_action_space_size = (
            self.action_type_size * self.source_actor_size * self.target_actor_size * self.param_size
        )
    
    # ========== Position/Actor Encoding ==========
    
    def position_to_target_index(self, x: int, y: int) -> int:
        """Convert board position (x, y) to target_actor index."""
        pos_idx = x * self.board_size + y
        return pos_idx + 1  # Offset by 1 (0 is None)
    
    def target_index_to_position(self, idx: int) -> Tuple[int, int]:
        """Convert target_actor index back to board position (x, y)."""
        pos_idx = idx - 1  # Remove offset
        x = pos_idx // self.board_size
        y = pos_idx % self.board_size
        return x, y
    
    def unit_id_to_source_index(self, unit_id: int) -> int:
        """Convert unit_id to source_actor index."""
        return unit_id  # unit_id directly in range [1..100]
    
    def unit_id_to_target_index(self, unit_id: int) -> int:
        """Convert unit_id to target_actor index (for targeting units)."""
        return unit_id + 121  # Offset to range [122..221]
    
    def city_id_to_source_index(self, city_id: int) -> int:
        """Convert city_id to source_actor index."""
        return city_id + 100  # Offset to range [101..150]
    
    def city_id_to_target_index(self, city_id: int) -> int:
        """Convert city_id to target_actor index (for targeting cities)."""
        return city_id + 221  # Offset to range [222..271]
    
    def tribe_id_to_target_index(self, tribe_id: int) -> int:
        """Convert tribe_id to target_actor index (for targeting tribes)."""
        return tribe_id + 271  # Offset to range [272..283]
    
    # ========== Component Index Encoding ==========
    
    def encode_action(
        self,
        action_type: str,
        source_actor: int = 0,
        target_actor: int = 0,
        param: int = 0,
    ) -> Dict[str, int]:
        """
        Encode an action into component indices.
        
        Args:
            action_type: Action type string (e.g., "MOVE")
            source_actor: Source actor index (0 = None)
            target_actor: Target actor index (0 = None)
            param: Parameter index
        
        Returns:
            Dict with keys ["action_type", "source_actor", "target_actor", "param"]
        """
        action_type_idx = self.action_type_str_to_idx.get(action_type)
        if action_type_idx is None:
            raise ValueError(f"Unknown action type: {action_type}")
        
        return {
            "action_type": action_type_idx,
            "source_actor": source_actor,
            "target_actor": target_actor,
            "param": param,
        }
    
    def decode_action(self, components: Dict[str, int]) -> Dict[str, Any]:
        """
        Decode component indices back to action description.
        
        Args:
            components: Dict with keys ["action_type", "source_actor", "target_actor", "param"]
        
        Returns:
            Dict with "action_type" string and component indices.
        """
        action_type_idx = components["action_type"]
        action_type_str = self.action_type_idx_to_str.get(action_type_idx, f"UNKNOWN_{action_type_idx}")
        
        return {
            "action_type": action_type_str,
            "action_type_idx": action_type_idx,
            **components,
        }
    
    # ========== Masking ==========
    
    def create_action_mask(
        self,
        available_actions: List[Dict[str, Any]],
        board_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create component-level masks for available actions.
        
        This is the key function for integrating with NN policy heads.
        
        Args:
            available_actions: List of dicts from game state with keys
                ["action_type", "class_name", "description", "index", "global_action_index"]
                (global_action_index is computed from encode_action for each available action)
            board_size: Optional override for board size (unused for now, but useful for validation)
        
        Returns:
            Tuple of (action_type_mask, source_mask, target_mask, param_mask)
            Each mask is binary [0, 1] where 1 = legal.
        """
        action_type_mask = np.zeros(self.action_type_size, dtype=np.float32)
        source_mask = np.zeros(self.source_actor_size, dtype=np.float32)
        target_mask = np.zeros(self.target_actor_size, dtype=np.float32)
        param_mask = np.zeros(self.param_size, dtype=np.float32)
        
        for action in available_actions:
            action_type_str = action.get("action_type")

            # Prefer structured encoded components from Java when present.
            if "encoded_components" in action:
                components = action["encoded_components"]
            else:
                components = self.encode_action(action_type_str)

            action_type_idx = components.get("action_type_index", components.get("action_type", 0))
            source_idx = components.get("source_actor_index", components.get("source_actor", 0))
            target_idx = components.get("target_actor_index", components.get("target_actor", 0))
            param_idx = components.get("param_index", components.get("param", 0))

            # Mark these indices as legal; zero is a valid index for "none".
            if 0 <= action_type_idx < self.action_type_size:
                action_type_mask[action_type_idx] = 1.0
            if 0 <= source_idx < self.source_actor_size:
                source_mask[source_idx] = 1.0
            if 0 <= target_idx < self.target_actor_size:
                target_mask[target_idx] = 1.0
            if 0 <= param_idx < self.param_size:
                param_mask[param_idx] = 1.0
        
        return action_type_mask, source_mask, target_mask, param_mask
    
    def mask_available_actions(
        self,
        available_actions: List[Dict[str, Any]],
    ) -> Dict[str, np.ndarray]:
        """
        Create masks for available actions. Returns a dict of component masks.
        
        Args:
            available_actions: List of action dicts from game state.
        
        Returns:
            Dict with keys ["action_type_mask", "source_mask", "target_mask", "param_mask"]
        """
        action_type_mask, source_mask, target_mask, param_mask = self.create_action_mask(available_actions)
        
        return {
            "action_type_mask": action_type_mask,
            "source_mask": source_mask,
            "target_mask": target_mask,
            "param_mask": param_mask,
        }
    
    # ========== Policy Composition (NN -> Action) ==========
    
    def sample_action_from_logits(
        self,
        action_type_logits: np.ndarray,
        source_logits: np.ndarray,
        target_logits: np.ndarray,
        param_logits: np.ndarray,
        available_actions: List[Dict[str, Any]],
        temperature: float = 1.0,
    ) -> Tuple[str, int]:
        """
        Sample an action given factorized logits and available actions.
        
        This demonstrates how to go from NN outputs back to a valid action.
        In practice, you may want more sophisticated sampling (e.g., top-k, nucleus sampling).
        
        Args:
            action_type_logits: [action_type_size]
            source_logits: [source_actor_size]
            target_logits: [target_actor_size]
            param_logits: [param_size]
            available_actions: List of legal actions from game state
            temperature: Softmax temperature
        
        Returns:
            (action_type_string, local_action_index_in_available_actions)
        """
        from scipy.special import softmax
        
        # Apply masks
        masks = self.mask_available_actions(available_actions)
        
        action_type_logits_masked = action_type_logits.copy()
        source_logits_masked = source_logits.copy()
        target_logits_masked = target_logits.copy()
        param_logits_masked = param_logits.copy()
        
        # Apply masks: set illegal indices to -inf
        action_type_logits_masked[masks["action_type_mask"] == 0] = -np.inf
        source_logits_masked[masks["source_mask"] == 0] = -np.inf
        target_logits_masked[masks["target_mask"] == 0] = -np.inf
        param_logits_masked[masks["param_mask"] == 0] = -np.inf
        
        # Convert logits to probabilities
        action_type_probs = softmax(action_type_logits_masked / temperature)
        source_probs = softmax(source_logits_masked / temperature)
        target_probs = softmax(target_logits_masked / temperature)
        param_probs = softmax(param_logits_masked / temperature)
        
        # Sample from each component
        action_type_idx = np.random.choice(self.action_type_size, p=action_type_probs)
        source_idx = np.random.choice(self.source_actor_size, p=source_probs)
        target_idx = np.random.choice(self.target_actor_size, p=target_probs)
        param_idx = np.random.choice(self.param_size, p=param_probs)
        
        sampled_components = {
            "action_type": action_type_idx,
            "source_actor": source_idx,
            "target_actor": target_idx,
            "param": param_idx,
        }
        
        # Find the closest available action
        best_action_idx = 0
        best_match_score = -1
        
        for i, action in enumerate(available_actions):
            action_type_str = action.get("action_type")
            if self.action_type_str_to_idx.get(action_type_str) == action_type_idx:
                best_action_idx = i
                break
        
        action_type_str = self.action_type_idx_to_str[action_type_idx]
        return action_type_str, best_action_idx
    
    # ========== Utility: Create Global Index Map ==========
    
    def create_global_action_index_map(
        self,
        available_actions: List[Dict[str, Any]],
        active_tribe_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Annotate available_actions with encoded_components (so they can be used for training).
        
        This is called by Java/captures to add global indices to each action.
        
        Args:
            available_actions: List of available action dicts
            active_tribe_id: Current active tribe ID (for context)
        
        Returns:
            Annotated list with "encoded_components" added to each action.
        """
        annotated = []
        for action in available_actions:
            action_copy = action.copy()
            
            # Parse action description to extract parameters
            # This is a heuristic parser; Java should ideally provide structured data.
            desc = action.get("description", "")
            action_type = action.get("action_type", "")
            
            # Default encoding
            components = self.encode_action(action_type, source_actor=0, target_actor=0, param=0)
            
            # Try to parse description to extract indices (very heuristic)
            # E.g., "MOVE by unit 2 to 3 : 8" -> unit=2, target pos=(3, 8)
            if action_type == "MOVE":
                parts = desc.split()
                try:
                    unit_idx = int(parts[3])
                    target_x = int(parts[5])
                    target_y = int(parts[7])
                    components["source_actor"] = self.unit_id_to_source_index(unit_idx)
                    components["target_actor"] = self.position_to_target_index(target_x, target_y)
                except (ValueError, IndexError):
                    pass
            
            # Add encoded components
            action_copy["encoded_components"] = components
            annotated.append(action_copy)
        
        return annotated


# Global instance for convenience
_encoder_instance = None


def get_encoder(schema_path: str = "action_space_schema.json") -> ActionSpaceEncoder:
    """Get or create the global encoder instance."""
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = ActionSpaceEncoder(schema_path)
    return _encoder_instance


if __name__ == "__main__":
    # Test the encoder
    encoder = ActionSpaceEncoder()
    
    print("Action Space Schema Loaded:")
    print(f"  Action types: {encoder.action_type_size}")
    print(f"  Source actors: {encoder.source_actor_size}")
    print(f"  Target actors: {encoder.target_actor_size}")
    print(f"  Params: {encoder.param_size}")
    print(f"  Total flattened space: {encoder.total_action_space_size:,}")
    
    # Test encoding
    move_encoding = encoder.encode_action("MOVE", source_actor=5, target_actor=45, param=0)
    print(f"\nMOVE encoding: {move_encoding}")
    
    # Test position conversion
    pos_idx = encoder.position_to_target_index(3, 5)
    print(f"Position (3, 5) -> target index {pos_idx}")
    x, y = encoder.target_index_to_position(pos_idx)
    print(f"Target index {pos_idx} -> position ({x}, {y})")
    
    # Test decoding
    decoded = encoder.decode_action(move_encoding)
    print(f"Decoded: {decoded}")
