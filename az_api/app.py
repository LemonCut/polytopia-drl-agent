import asyncio
import json
import os
import uuid
import numpy as np
import torch
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from action_encoding import ActionSpaceEncoder
from model import TribesModel, StateEncoder, encode_state, TribesTransformerModel, env_bool

app = FastAPI()


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Invalid integer env {name}={value}; using {default}")
        return default


INFERENCE_CONCURRENCY = max(1, env_int("TRIBES_INFERENCE_CONCURRENCY", 1))
QUERY_SEMAPHORE = asyncio.Semaphore(INFERENCE_CONCURRENCY)

CAPTURE_DIR = Path("captures")
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# /query can be called extremely frequently by MCTS (tens of thousands of times per game).
# Logging every inference request to disk will quickly create huge numbers of files and slow runs.
# Enable this only when you explicitly want to debug payload contents.
SAVE_INFERENCE_REQUESTS = os.environ.get("TRIBES_SAVE_INFERENCE", "0").strip().lower() in {"1", "true", "yes"}
INFERENCE_DIR = Path(os.environ.get("TRIBES_INFERENCE_DIR", "inference"))
if SAVE_INFERENCE_REQUESTS:
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

MASK_SEND_STARS = env_bool("TRIBES_MASK_SEND_STARS", True)

# Load the action space encoder
encoder = ActionSpaceEncoder()

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# Load the PyTorch model
state_encoder = StateEncoder()
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TORCH_THREADS = env_int("TRIBES_TORCH_THREADS", 0)
if TORCH_THREADS > 0:
    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(max(1, min(2, TORCH_THREADS)))
    except RuntimeError as exc:
        print(f"Could not set Torch interop threads: {exc}")

torch.backends.cudnn.benchmark = True


model = TribesTransformerModel(
    state_size=state_encoder.total_state_size,
    mask_send_stars=MASK_SEND_STARS,
).to(device)

model.eval()



# Try to load model weights if they exist
MODEL_PATH = Path(os.environ.get("TRIBES_MODEL_PATH", "model_weights.pth"))
if MODEL_PATH.exists():
    try:
        model.load(str(MODEL_PATH), device=str(device))
        print(f"Loaded model weights from {MODEL_PATH}")
    except Exception as exc:
        print(f"Failed to load model weights from {MODEL_PATH}: {exc}")
        print("Using untrained model with current action space sizes")
else:
    print(f"No model weights found at {MODEL_PATH}, using untrained model")

model = model.to(device)


def effective_available_actions(available_actions):
    if not MASK_SEND_STARS:
        return available_actions
    return [
        action
        for action in available_actions
        if action.get("action_type") != "SEND_STARS"
    ]


def write_json_atomic(output_path: Path, payload: dict) -> None:
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)
    temp_path.replace(output_path)


@app.get("/hello")
async def hello():
    return {"message": "hello from python"}


@app.post("/capture")
async def capture(req: Request):
    payload = await req.json()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    tick = payload.get("tick", "unknown")
    action_count = payload.get("available_action_count", "na")
    policy_type = payload.get("policy_type", "capture")
    filename = (
        f"{policy_type}_tick{tick}_actions{action_count}_"
        f"{timestamp}_{uuid.uuid4().hex[:8]}.json"
    )
    output_path = CAPTURE_DIR / filename

    write_json_atomic(output_path, payload)

    return {
        "status": "captured",
        "captured_path": str(output_path),
        "tick": tick,
        "policy_type": policy_type,
    }


@app.post("/result")
async def result(req: Request):
    payload = await req.json()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    game_seed = payload.get("game_seed", "unknown")
    player_id = payload.get("player_id", "na")
    filename = (
        f"result_game{game_seed}_player{player_id}_"
        f"{timestamp}_{uuid.uuid4().hex[:8]}.json"
    )
    output_path = RESULTS_DIR / filename

    write_json_atomic(output_path, payload)

    return {
        "status": "saved",
        "result_path": str(output_path),
        "game_seed": game_seed,
        "player_id": player_id,
    }


@app.post("/query")
async def query(req: Request):
    """
    Receive game state and available actions from Java.
    Return a policy (action logits) over available actions using the trained model.
    
    The model encodes the state exactly as Java sends it. AlphaZero training uses
    full-observability payloads; partial-observation experiments must send already
    redacted payloads.
    """
    payload = await req.json()
    async with QUERY_SEMAPHORE:
        return await asyncio.to_thread(build_policy_response, payload)


def build_policy_response(payload):
    tick = payload.get("tick", "unknown")
    action_count = payload.get("available_action_count", "na")

    output_path = None
    if SAVE_INFERENCE_REQUESTS:
        # Save the request for analysis (kept separate from training captures)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        filename = (
            f"inference_tick{tick}_actions{action_count}_"
            f"{timestamp}_{uuid.uuid4().hex[:8]}.json"
        )
        output_path = INFERENCE_DIR / filename
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, sort_keys=True)

    # Extract game state
    available_actions = payload.get("available_actions", [])
    policy_available_actions = effective_available_actions(available_actions)

    # Create masks for legal actions
    try:
        masks = encoder.mask_available_actions(policy_available_actions)
        
        # Encode the state and get model predictions
        state_tensor = encode_state(payload, state_encoder)
        
        with torch.inference_mode():
            state_batch = state_tensor.unsqueeze(0).to(device)  # Add batch dimension
            action_type_logits, source_logits, target_logits, param_logits, value_pred = model(state_batch)
            
            # Remove batch dimension and convert to numpy
            action_type_logits = action_type_logits.squeeze(0).cpu().numpy()
            source_logits = source_logits.squeeze(0).cpu().numpy()
            target_logits = target_logits.squeeze(0).cpu().numpy()
            param_logits = param_logits.squeeze(0).cpu().numpy()
            # Value is returned from the perspective of the active player.
            value = torch.tanh(value_pred.squeeze(0).squeeze(-1)).cpu().item()
        
        # Compute masked softmax probabilities (post-softmax masked entries will be 0)
        def masked_softmax(logits, mask):
            # logits and mask are numpy arrays
            masked_logits = np.where(np.array(mask) > 0, logits, -1e9)
            maxv = np.max(masked_logits)
            exps = np.exp(masked_logits - maxv)
            exps = exps * (np.array(mask) > 0)
            s = np.sum(exps)
            if s == 0:
                # no allowed entries -> uniform over mask==1 (or uniform over all if none)
                allowed = np.sum(np.array(mask) > 0)
                if allowed == 0:
                    return np.ones_like(exps) / len(exps)
                return (np.array(mask) > 0).astype(float) / allowed
            return exps / s

        action_type_probs = masked_softmax(action_type_logits, masks["action_type_mask"])
        source_probs = masked_softmax(source_logits, masks["source_mask"])
        target_probs = masked_softmax(target_logits, masks["target_mask"])
        param_probs = masked_softmax(param_logits, masks["param_mask"])

        def json_safe_logits(logits):
            return np.nan_to_num(logits, neginf=-1e9, posinf=1e9).tolist()
        
        policy_response = {
            "status": "success",
            "policy_type": "neural_network",
            "value": float(value),
            "mask_send_stars": MASK_SEND_STARS,
            "action_type_logits": json_safe_logits(action_type_logits),
            "action_type_probs": action_type_probs.tolist(),
            "source_logits": json_safe_logits(source_logits),
            "source_probs": source_probs.tolist(),
            "target_logits": json_safe_logits(target_logits),
            "target_probs": target_probs.tolist(),
            "param_logits": json_safe_logits(param_logits),
            "param_probs": param_probs.tolist(),
            "masks": {
                "action_type_mask": masks["action_type_mask"].tolist(),
                "source_mask": masks["source_mask"].tolist(),
                "target_mask": masks["target_mask"].tolist(),
                "param_mask": masks["param_mask"].tolist(),
            },
            "num_legal_actions": len(available_actions),
            "captured_path": str(output_path) if output_path else None,
            "tick": tick,
        }
    except Exception as e:
        # If encoding fails, return error response
        import traceback
        policy_response = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "captured_path": str(output_path) if output_path else None,
            "tick": tick,
        }

    return policy_response
