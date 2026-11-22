"""E5: Local Reference + Latent Residual Learning built on official BPP components."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E5_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = E5_ROOT / "outputs"

BPP_SRC = ROOT / "bpp/src"
BPP_CKPT = ROOT / "bpp/checkpoints/liberogen_goal_chain_behavior_prompting.ckpt"
RAW_LIBERO = ROOT / "bpp/datasets/libero"

E4_OUTPUTS = ROOT / "e4_chain/outputs"
MANIFEST_80_DIR = (
    E4_OUTPUTS / "spatial_8x8/query_windows/bpp/seed42/277e1c5c8129e89c"
)

CAMERAS = ("agentview_rgb", "eye_in_hand_rgb")
IMAGE_SIZE = 224
CENTER_CROP_RATIO = 0.95
DEMO_STRIDE = 20  # 20Hz -> 1Hz (BPP prompt_chunk_n_actions=20)
MAX_DEMO_LEN = 50  # BPP max_sequence_length=1000 / 20 = 50
QUERY_HISTORY = 2  # BPP img_obs_horizon=2
ACTION_HORIZON = 16  # BPP action_horizon=16
EXEC_HORIZON = 12  # BPP exec_action_horizon=12
ACTION_DIM_10D = 10
ACTION_DIM_7D = 7
EMBED_DIM = 768
NUM_HEADS = 8
NUM_ATTENTION_SINKS = 4
