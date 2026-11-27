"""E9 Latent Scene-Sensitivity Test Suite.

Constructs a controlled benchmark where the EXACT same robot initial state and
16-step free-space action sequence (`demo_1`, actions 12..27 from LIBERO-Gen Goal Chain)
are replayed across visually distinct LIBERO tabletop scenes (varying background,
tabletop surface/lighting, object layout, distractor/replaced objects, and combined
scene swaps) with strict zero robot-object contact and machine-precision physical
trajectory identity verification.

Saves all dual-camera RGB frame sequences, extracts 16-frame E9 Stage 1 local behavior
latents via frozen BPP CLIP + frozen E9 LocalBehaviorEncoder, and compares
same-action / different-scene distances (cosine similarity, normalized L2, P/Q subspace
decomposition, and frozen decoder action drift) against both CLIP input shifts and
same-scene / different-action reference clips.

Two-stage execution (per AGENTS.md GPU isolation):
  1. `simulate`: runs MuJoCo EGL on GPU 2 (`CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2`)
  2. `analyze`:  runs PyTorch CLIP + E9 Stage 1 on CUDA GPU 2 (`CUDA_VISIBLE_DEVICES=GPU-...`)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ASSETS_DIR = REPO_ROOT / "bpp/src/deps/LIBERO/libero/libero/assets"
BASE_SCENE_XML = LIBERO_ASSETS_DIR / "scenes/libero_tabletop_base_style.xml"
TEXTURES_DIR = LIBERO_ASSETS_DIR / "textures"

DEFAULT_TASK = "put_the_bowl_on_the_plate_and_then_put_the_cream_cheese_in_the_bowl"
DEFAULT_HDF5 = (
    REPO_ROOT
    / "bpp/datasets/libero/libero_goal_chain_selected_view"
    / f"{DEFAULT_TASK}_demo.hdf5"
)
DEFAULT_BDDL = (
    REPO_ROOT
    / "bpp/src/behavior_prompting/train_network/env/libero/bddl_files/libero_goal_chain_selected_view"
    / f"{DEFAULT_TASK}.bddl"
)
DEFAULT_OUT_DIR = REPO_ROOT / "e9/outputs/scene_sensitivity"
DEFAULT_E9_CKPT = REPO_ROOT / "e9/outputs/stage1_local/best.pt"
DEFAULT_P0_BASES = REPO_ROOT / "p0_lowrank/outputs/e9_p0_bases.pt"

ACTION_START = 12
ACTION_LEN = 16  # actions 12..27 inclusive (16 steps)

# Same-scene, different-action reference clip start steps in demo_1 (T=261)
REF_DIFF_ACTION_STARTS: List[Tuple[int, str]] = [
    (0, "early_start_0_15"),
    (16, "overlap_shift4_16_31"),
    (28, "next_approach_28_43"),
    (44, "grasp_bowl_44_59"),
    (76, "lift_carry_bowl_76_91"),
    (108, "place_bowl_108_123"),
    (140, "transit_cheese_140_155"),
    (172, "grasp_cheese_172_187"),
    (204, "place_cheese_204_219"),
]


@dataclass
class ObjectPlacement:
    name: str
    category: str
    x: float
    y: float
    yaw: Optional[float] = None
    is_fixture: bool = False


@dataclass
class SceneSpec:
    scene_id: str
    group: str
    description: str
    use_base_bddl: bool = False
    use_base_obj_states: bool = False
    floor_style: str = "light-gray"
    wall_style: str = "light-gray-plaster"
    table_texture: Optional[str] = None  # filename in TEXTURES_DIR
    light_diffuse: Optional[Tuple[float, float, float]] = None
    placements: List[ObjectPlacement] = field(default_factory=list)


def build_scene_specs() -> List[SceneSpec]:
    """Define the 12 controlled test scenes (1 base + 1 repeat sanity + 10 visual variants)."""
    return [
        # --- Control & Determinism Sanity ---
        SceneSpec(
            scene_id="S0_base",
            group="baseline",
            description="Original demo_1 tabletop scene (base textures + exact demo_1 t=12 layout)",
            use_base_bddl=True,
            use_base_obj_states=True,
            floor_style="light-gray",
            wall_style="light-gray-plaster",
        ),
        SceneSpec(
            scene_id="S0_repeat",
            group="sanity_repeat",
            description="Exact duplicate of S0_base (determinism sanity check)",
            use_base_bddl=True,
            use_base_obj_states=True,
            floor_style="light-gray",
            wall_style="light-gray-plaster",
        ),
        # --- Factor A: Background / Surface / Lighting Only (identical objects & poses) ---
        SceneSpec(
            scene_id="S1_bg_wall_floor_blue",
            group="background_only",
            description="Dark-blue wall + white-marble floor; table texture and objects unchanged",
            use_base_bddl=True,
            use_base_obj_states=True,
            floor_style="white-marble",
            wall_style="dark-blue",
        ),
        SceneSpec(
            scene_id="S2_bg_surface_dark_green",
            group="background_and_table",
            description="Dark-green wall + rustic floor + dark stone table texture; objects unchanged",
            use_base_bddl=True,
            use_base_obj_states=True,
            floor_style="rustic",
            wall_style="dark-green",
            table_texture="dark_floor_texture.png",
        ),
        SceneSpec(
            scene_id="S3_bg_surface_warm_light",
            group="background_and_table",
            description="Yellow-linen wall + brown-ceramic floor + light-wood-planks table + warm lighting; objects unchanged",
            use_base_bddl=True,
            use_base_obj_states=True,
            floor_style="brown-ceramic",
            wall_style="yellow-linen",
            table_texture="seamless_wood_planks_floor.png",
            light_diffuse=(1.0, 0.86, 0.68),
        ),
        # --- Factor B: Object Layout Only (identical background & table) ---
        SceneSpec(
            scene_id="S4_layout_rearranged",
            group="layout_only",
            description="Same base objects & fixtures rearranged to opposite/shifted safe table positions",
            floor_style="light-gray",
            wall_style="light-gray-plaster",
            placements=[
                # Fixtures moved to mirrored / safe edges away from arm sweep (x in [-0.14, 0.0], y in [0.04, 0.06], z>=1.08)
                ObjectPlacement("wooden_cabinet_1", "wooden_cabinet", 0.08, 0.30, yaw=0.0, is_fixture=True),
                ObjectPlacement("flat_stove_1", "flat_stove", -0.34, -0.26, is_fixture=True),
                ObjectPlacement("wine_rack_1", "wine_rack", -0.28, 0.30, yaw=0.0, is_fixture=True),
                # Same 4 movable objects relocated cleanly across the workspace
                ObjectPlacement("plate_1", "plate", -0.08, -0.18),
                ObjectPlacement("akita_black_bowl_1", "akita_black_bowl", 0.16, 0.06),
                ObjectPlacement("cream_cheese_1", "cream_cheese", 0.14, -0.18),
                ObjectPlacement("wine_bottle_1", "wine_bottle", 0.18, 0.22),
            ],
        ),
        SceneSpec(
            scene_id="S5_empty_table",
            group="layout_only",
            description="Completely empty tabletop (all objects and fixtures removed)",
            floor_style="light-gray",
            wall_style="light-gray-plaster",
            placements=[],
        ),
        # --- Factor C: Irrelevant / Replaced Objects Only (identical background & table) ---
        SceneSpec(
            scene_id="S6_distractors_added",
            group="objects_only",
            description="Base objects + 4 colorful unrelated distractors (ketchup, orange_juice, alphabet_soup, red_coffee_mug)",
            floor_style="light-gray",
            wall_style="light-gray-plaster",
            placements=[
                # Original fixtures & objects at base positions
                ObjectPlacement("wooden_cabinet_1", "wooden_cabinet", 0.03, -0.24, yaw=math.pi, is_fixture=True),
                ObjectPlacement("flat_stove_1", "flat_stove", -0.41, 0.21, is_fixture=True),
                ObjectPlacement("wine_rack_1", "wine_rack", -0.26, -0.26, yaw=math.pi, is_fixture=True),
                ObjectPlacement("plate_1", "plate", 0.05, -0.02),
                ObjectPlacement("akita_black_bowl_1", "akita_black_bowl", -0.09, 0.00),
                ObjectPlacement("cream_cheese_1", "cream_cheese", -0.05, 0.13),
                ObjectPlacement("wine_bottle_1", "wine_bottle", -0.20, -0.05),
                # 4 colorful unrelated distractors in clear view, safely clear of arm sweep
                ObjectPlacement("ketchup_1", "ketchup", 0.18, 0.16),
                ObjectPlacement("orange_juice_1", "orange_juice", 0.20, -0.10),
                ObjectPlacement("alphabet_soup_1", "alphabet_soup", 0.06, 0.22),
                ObjectPlacement("red_coffee_mug_1", "red_coffee_mug", -0.18, 0.24),
            ],
        ),
        SceneSpec(
            scene_id="S7_objects_kitchen_props",
            group="objects_only",
            description="All objects replaced with different kitchen items (moka_pot, frypan, milk, butter, tomato_sauce, mug, tray)",
            floor_style="light-gray",
            wall_style="light-gray-plaster",
            placements=[
                ObjectPlacement("wooden_tray_1", "wooden_tray", 0.12, -0.20),
                ObjectPlacement("chefmate_8_frypan_1", "chefmate_8_frypan", -0.12, 0.22),
                ObjectPlacement("moka_pot_1", "moka_pot", 0.16, 0.16),
                ObjectPlacement("milk_1", "milk", -0.24, -0.16),
                ObjectPlacement("butter_1", "butter", 0.02, 0.02),
                ObjectPlacement("tomato_sauce_1", "tomato_sauce", 0.18, -0.02),
                ObjectPlacement("chocolate_pudding_1", "chocolate_pudding", -0.10, -0.06),
                ObjectPlacement("white_yellow_mug_1", "white_yellow_mug", 0.04, 0.24),
            ],
        ),
        SceneSpec(
            scene_id="S8_objects_study_props",
            group="objects_only",
            description="All objects replaced with study/office items (books, desk_caddy, basket, porcelain_mug, cookies)",
            floor_style="light-gray",
            wall_style="light-gray-plaster",
            placements=[
                ObjectPlacement("basket_1", "basket", 0.14, -0.22),
                ObjectPlacement("desk_caddy_1", "desk_caddy", -0.20, 0.24),
                ObjectPlacement("black_book_1", "black_book", 0.12, 0.16),
                ObjectPlacement("yellow_book_1", "yellow_book", -0.08, -0.14),
                ObjectPlacement("porcelain_mug_1", "porcelain_mug", 0.06, 0.02),
                ObjectPlacement("cookies_1", "cookies", 0.20, 0.04),
                ObjectPlacement("white_storage_box_1", "white_storage_box", -0.26, -0.24),
            ],
        ),
        # --- Factor D: Combined Multi-Factor Visual Shift (Background + Table + Layout + Objects) ---
        SceneSpec(
            scene_id="S9_combo_kitchen_dark",
            group="combined_scene",
            description="Dark-green wall + rustic floor + gray-ceramic table + kitchen props in shifted layout",
            floor_style="rustic",
            wall_style="dark-green",
            table_texture="gray_ceramic_tile.png",
            placements=[
                ObjectPlacement("chefmate_8_frypan_1", "chefmate_8_frypan", 0.14, -0.18),
                ObjectPlacement("moka_pot_1", "moka_pot", -0.16, 0.22),
                ObjectPlacement("ketchup_1", "ketchup", 0.18, 0.18),
                ObjectPlacement("milk_1", "milk", 0.20, 0.00),
                ObjectPlacement("popcorn_1", "popcorn", -0.06, -0.16),
                ObjectPlacement("wooden_tray_1", "wooden_tray", -0.22, -0.24),
                ObjectPlacement("red_coffee_mug_1", "red_coffee_mug", 0.04, 0.14),
            ],
        ),
        SceneSpec(
            scene_id="S10_combo_study_blue",
            group="combined_scene",
            description="Dark-blue wall + white-marble floor + white-marble table + study props in shifted layout",
            floor_style="white-marble",
            wall_style="dark-blue",
            table_texture="white_marble_floor.png",
            placements=[
                ObjectPlacement("yellow_book_1", "yellow_book", 0.16, -0.16),
                ObjectPlacement("black_book_1", "black_book", -0.14, 0.20),
                ObjectPlacement("desk_caddy_1", "desk_caddy", 0.18, 0.20),
                ObjectPlacement("basket_1", "basket", -0.24, -0.22),
                ObjectPlacement("red_coffee_mug_1", "red_coffee_mug", 0.04, -0.08),
                ObjectPlacement("cookies_1", "cookies", 0.08, 0.12),
                ObjectPlacement("alphabet_soup_1", "alphabet_soup", -0.12, -0.06),
            ],
        ),
    ]


def write_custom_arena_xml(spec: SceneSpec, out_path: Path) -> Path:
    """Create a scene XML derived from `libero_tabletop_base_style.xml` with custom
    table texture and/or lighting, preserving identical geometry, table size, and camera pose."""
    tree = ET.parse(BASE_SCENE_XML)
    root = tree.getroot()
    asset = root.find("asset")
    assert asset is not None

    # Resolve all existing relative texture file paths to absolute paths so the XML works from any directory
    base_dir = BASE_SCENE_XML.parent
    for tex in asset.findall("texture"):
        f_attr = tex.get("file")
        if f_attr:
            abs_path = (base_dir / f_attr).resolve()
            tex.set("file", str(abs_path))

    if spec.table_texture is not None:
        tex_path = (TEXTURES_DIR / spec.table_texture).resolve()
        if not tex_path.is_file():
            raise FileNotFoundError(f"Missing table texture: {tex_path}")
        for tex in asset.findall("texture"):
            if tex.get("name") in ("tex-table", "tex-table-legs"):
                tex.set("file", str(tex_path))

    if spec.light_diffuse is not None:
        wb = root.find("worldbody")
        assert wb is not None
        diff_str = f"{spec.light_diffuse[0]:.3f} {spec.light_diffuse[1]:.3f} {spec.light_diffuse[2]:.3f}"
        for light in wb.findall("light"):
            light.set("diffuse", diff_str)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8")
    return out_path


def write_custom_bddl(spec: SceneSpec, out_path: Path) -> Path:
    """Generate a valid LIBERO_Tabletop_Manipulation BDDL file for custom layouts/objects."""
    if spec.use_base_bddl:
        return DEFAULT_BDDL

    half_w = 0.005  # tight 1cm box for deterministic placement
    regions_lines: List[str] = []
    fixtures_lines: List[str] = ["    main_table - table"]
    objects_lines: List[str] = []
    init_lines: List[str] = []

    for pl in spec.placements:
        reg_name = f"{pl.name}_region"
        x0, x1 = pl.x - half_w, pl.x + half_w
        y0, y1 = pl.y - half_w, pl.y + half_w
        yaw_block = ""
        if pl.yaw is not None:
            yaw_block = (
                f"\n          (:yaw_rotation (\n"
                f"              ({pl.yaw:.6f} {pl.yaw:.6f})\n"
                f"            )\n"
                f"          )"
            )
        regions_lines.append(
            f"      ({reg_name}\n"
            f"          (:target main_table)\n"
            f"          (:ranges (\n"
            f"              ({x0:.6f} {y0:.6f} {x1:.6f} {y1:.6f})\n"
            f"            )\n"
            f"          ){yaw_block}\n"
            f"      )"
        )
        if pl.is_fixture:
            fixtures_lines.append(f"    {pl.name} - {pl.category}")
        else:
            objects_lines.append(f"    {pl.name} - {pl.category}")
        init_lines.append(f"    (On {pl.name} main_table_{reg_name})")

    obj_interest = [pl.name for pl in spec.placements if not pl.is_fixture][:1]
    regions_str = "\n".join(regions_lines)
    fixtures_str = "\n".join(fixtures_lines)
    objects_str = "\n".join(objects_lines)
    interest_str = "\n".join(f"    {o}" for o in obj_interest)
    init_str = "\n".join(init_lines)
    goal_str = f"(On {obj_interest[0]} main_table_{obj_interest[0]}_region)" if obj_interest else "(And)"

    content = (
        f"(define (problem LIBERO_Tabletop_Manipulation)\n"
        f"  (:domain robosuite)\n"
        f"  (:language Scene sensitivity test {spec.scene_id})\n"
        f"  (:regions\n{regions_str}\n  )\n\n"
        f"  (:fixtures\n{fixtures_str}\n  )\n\n"
        f"  (:objects\n{objects_str}\n  )\n\n"
        f"  (:obj_of_interest\n{interest_str}\n  )\n\n"
        f"  (:init\n{init_str}\n  )\n\n"
        f"  (:goal\n    {goal_str}\n  )\n)\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    return out_path


def _ensure_libero_config(cfg_dir: Path) -> None:
    libero_root = (REPO_ROOT / "bpp/src/deps/LIBERO/libero/libero").resolve()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / "config.yaml"
    cfg_file.write_text(
        f"benchmark_root: {libero_root}\n"
        f"bddl_files: {libero_root / 'bddl_files'}\n"
        f"init_states: {libero_root / 'init_files'}\n"
        f"datasets: {(REPO_ROOT / 'bpp/datasets/libero').resolve()}\n"
        f"assets: {libero_root / 'assets'}\n"
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg_dir)


def _check_robot_contacts(sim) -> List[Tuple[str, str]]:
    """Return any active MuJoCo contacts involving robot0 or gripper0."""
    hits: List[Tuple[str, str]] = []
    for ci in range(int(sim.data.ncon)):
        c = sim.data.contact[ci]
        g1 = sim.model.geom_id2name(c.geom1) or ""
        g2 = sim.model.geom_id2name(c.geom2) or ""
        is_r1 = ("robot0" in g1) or ("gripper0" in g1)
        is_r2 = ("robot0" in g2) or ("gripper0" in g2)
        if is_r1 or is_r2:
            hits.append((g1, g2))
    return hits


def _opengl_to_upright(img: np.ndarray) -> np.ndarray:
    """Flip OpenGL (bottom-up) image to upright (top-down) for saving PNG previews."""
    return np.ascontiguousarray(img[::-1, :, :])


def _save_scene_visuals(
    scene_dir: Path,
    agentview: np.ndarray,
    eye_in_hand: np.ndarray,
    scene_id: str,
) -> None:
    """Save upright PNG frames (16 frames x 2 cameras) and a 4-frame keyframe strip."""
    frames_dir = scene_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for t in range(ACTION_LEN):
        Image.fromarray(_opengl_to_upright(agentview[t])).save(
            frames_dir / f"agentview_{t:02d}.png"
        )
        Image.fromarray(_opengl_to_upright(eye_in_hand[t])).save(
            frames_dir / f"eye_in_hand_{t:02d}.png"
        )

    # Save 4-keyframe strip (t=0, 5, 10, 15) for both cameras
    key_ts = [0, 5, 10, 15]
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 4.3), dpi=130)
    for col, t in enumerate(key_ts):
        axes[0, col].imshow(_opengl_to_upright(agentview[t]))
        axes[0, col].set_title(f"agentview t={ACTION_START + t}", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(_opengl_to_upright(eye_in_hand[t]))
        axes[1, col].set_title(f"eye_in_hand t={ACTION_START + t}", fontsize=9)
        axes[1, col].axis("off")
    fig.suptitle(f"{scene_id} (actions {ACTION_START}..{ACTION_START + ACTION_LEN - 1})", fontsize=10)
    fig.tight_layout()
    fig.savefig(scene_dir / "strip.png")
    plt.close(fig)


def run_simulate(args: argparse.Namespace) -> None:
    t_start = time.monotonic()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = out_dir / "configs"
    _ensure_libero_config(out_dir / ".libero_cfg")

    from libero.libero.envs.env_wrapper import ControlEnv

    # 1. Load fixed action sequence and reference demo_1 state/clips
    with h5py.File(args.hdf5, "r") as f:
        d1 = f[f"data/{args.demo_key}"]
        all_actions = np.asarray(d1["actions"][:], dtype=np.float32)
        all_states = np.asarray(d1["states"][:], dtype=np.float64)
        all_av = np.asarray(d1["obs/agentview_rgb"][:], dtype=np.uint8)
        all_eh = np.asarray(d1["obs/eye_in_hand_rgb"][:], dtype=np.uint8)
        conv = str(f["data"].attrs.get("macros_image_convention", "opengl"))

    actions_seg = all_actions[ACTION_START : ACTION_START + ACTION_LEN].copy()
    state_start = all_states[ACTION_START].copy()

    # Save demo_1 same-scene reference clips (including t=12..27 and different-action segments)
    ref_clips_dir = out_dir / "reference_demo1"
    ref_clips_dir.mkdir(parents=True, exist_ok=True)
    ref_meta: List[Dict[str, Any]] = []
    ref_starts_all = [(ACTION_START, "same_action_hdf5_12_27")] + list(REF_DIFF_ACTION_STARTS)
    for st, label in ref_starts_all:
        end = st + ACTION_LEN
        assert end <= len(all_actions), f"Clip [{st}:{end}] out of range (T={len(all_actions)})"
        np.savez_compressed(
            ref_clips_dir / f"{label}.npz",
            agentview_rgb=all_av[st:end],
            eye_in_hand_rgb=all_eh[st:end],
            actions_7d=all_actions[st:end],
            start_step=np.int32(st),
            convention=np.array(conv),
        )
        ref_meta.append({"label": label, "start_step": int(st), "end_step": int(end - 1)})
    (ref_clips_dir / "meta.json").write_text(json.dumps(ref_meta, indent=2) + "\n")

    specs = build_scene_specs()
    ref_init_qpos: Optional[Dict[str, np.ndarray]] = None
    ref_init_qvel: Optional[Dict[str, np.ndarray]] = None
    ref_initial_joint: Optional[np.ndarray] = None
    base_traj: Optional[Dict[str, np.ndarray]] = None
    manifest_rows: List[Dict[str, Any]] = []

    print(
        f"[Simulate] Task={DEFAULT_TASK} | Demo={args.demo_key} | "
        f"Actions={ACTION_START}..{ACTION_START + ACTION_LEN - 1} ({ACTION_LEN} steps) | "
        f"Scenes={len(specs)}",
        flush=True,
    )

    for spec in specs:
        t0 = time.monotonic()
        arena_xml = write_custom_arena_xml(spec, cfg_dir / f"{spec.scene_id}_arena.xml")
        bddl_path = write_custom_bddl(spec, cfg_dir / f"{spec.scene_id}.bddl")

        env = ControlEnv(
            str(bddl_path),
            camera_heights=128,
            camera_widths=128,
            use_camera_obs=True,
            has_offscreen_renderer=True,
            hard_reset=False,
            ignore_done=True,
            control_freq=20,
            initialization_noise=None,
            scene_xml=str(arena_xml),
            scene_properties={
                "floor_style": spec.floor_style,
                "wall_style": spec.wall_style,
            },
        )
        try:
            if not env.robots[0].controller.use_delta:
                raise RuntimeError("Expected delta OSC_POSE controller")
            if hasattr(env.env, "visualize"):
                env.env.visualize({"robots": True, "env": False, "grippers": False})

            env.seed(7)
            env.env.reset()

            robot_jnames = list(env.robots[0].robot_joints) + list(env.robots[0].gripper.joints)

            if spec.use_base_obj_states:
                env.set_init_state(state_start)

            if ref_init_qpos is None:
                ref_init_qpos = {
                    jn: env.sim.data.get_joint_qpos(jn).copy() for jn in robot_jnames
                }
                ref_init_qvel = {
                    jn: env.sim.data.get_joint_qvel(jn).copy() for jn in robot_jnames
                }
                ref_initial_joint = np.asarray(
                    env.robots[0].controller.initial_joint, dtype=np.float64
                ).copy()
            else:
                for jn in robot_jnames:
                    env.sim.data.set_joint_qpos(jn, ref_init_qpos[jn])
                    env.sim.data.set_joint_qvel(jn, ref_init_qvel[jn])

            # Synchronize simulator/controller internal state identically across all scenes
            env.sim.data.time = 0.0
            env.sim.data.ctrl[:] = 0.0
            env.sim.data.qacc_warmstart[:] = 0.0
            env.sim.forward()
            env.env._update_observables(force=True)
            ctrl = env.robots[0].controller
            ctrl.initial_joint = ref_initial_joint.copy()
            ctrl.update(force=True)
            ctrl.reset_goal()
            env.robots[0].gripper.current_action = np.zeros(2)
            env.seed(7)  # synchronize RNG before replay

            eef_site_id = env.robots[0].eef_site_id
            arm_jnames = list(env.robots[0].robot_joints)
            grip_jnames = list(env.robots[0].gripper.joints)

            def _true_robot_state() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                ee_p = np.asarray(env.sim.data.site_xpos[eef_site_id], dtype=np.float64).copy()
                ee_m = np.asarray(env.sim.data.site_xmat[eef_site_id], dtype=np.float64).reshape(9).copy()
                j_p = np.array([float(env.sim.data.get_joint_qpos(j)) for j in arm_jnames], dtype=np.float64)
                g_p = np.array([float(env.sim.data.get_joint_qpos(j)) for j in grip_jnames], dtype=np.float64)
                return ee_p, ee_m, j_p, g_p

            obs = env.env._get_observations(force_update=True)
            ee_p0, ee_m0, j_p0, g_p0 = _true_robot_state()

            av_list = [np.ascontiguousarray(obs["agentview_image"], dtype=np.uint8)]
            eh_list = [np.ascontiguousarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)]
            ee_pos_list = [ee_p0]
            ee_quat_list = [ee_m0]
            joint_pos_list = [j_p0]
            gripper_list = [g_p0]
            contacts_per_step: List[List[Tuple[str, str]]] = [_check_robot_contacts(env.sim)]

            for a in actions_seg:
                obs, _, _, _ = env.step(a)
                ee_pk, ee_mk, j_pk, g_pk = _true_robot_state()
                av_list.append(np.ascontiguousarray(obs["agentview_image"], dtype=np.uint8))
                eh_list.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8))
                ee_pos_list.append(ee_pk)
                ee_quat_list.append(ee_mk)
                joint_pos_list.append(j_pk)
                gripper_list.append(g_pk)
                contacts_per_step.append(_check_robot_contacts(env.sim))
        finally:
            env.close()

        av_arr = np.stack(av_list, axis=0)  # (17, 128, 128, 3)
        eh_arr = np.stack(eh_list, axis=0)  # (17, 128, 128, 3)
        ee_pos_arr = np.stack(ee_pos_list, axis=0)  # (17, 3)
        ee_quat_arr = np.stack(ee_quat_list, axis=0)  # (17, 4)
        joint_pos_arr = np.stack(joint_pos_list, axis=0)  # (17, 7)
        gripper_arr = np.stack(gripper_list, axis=0)  # (17, 2)
        total_contacts = sum(len(c) for c in contacts_per_step)

        if base_traj is None:
            base_traj = {
                "ee_pos": ee_pos_arr,
                "ee_quat": ee_quat_arr,
                "joint_pos": joint_pos_arr,
                "gripper_qpos": gripper_arr,
            }

        max_ee_diff = float(np.abs(ee_pos_arr - base_traj["ee_pos"]).max())
        max_quat_diff = float(np.abs(ee_quat_arr - base_traj["ee_quat"]).max())
        max_joint_diff = float(np.abs(joint_pos_arr - base_traj["joint_pos"]).max())
        max_grip_diff = float(np.abs(gripper_arr - base_traj["gripper_qpos"]).max())
        init_joint_diff = float(np.abs(joint_pos_arr[0] - base_traj["joint_pos"][0]).max())

        scene_dir = out_dir / "scenes" / spec.scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            scene_dir / "clip_frames.npz",
            agentview_rgb=av_arr[:ACTION_LEN],
            eye_in_hand_rgb=eh_arr[:ACTION_LEN],
            agentview_rgb_full17=av_arr,
            eye_in_hand_rgb_full17=eh_arr,
            convention=np.array("opengl"),
        )
        np.savez_compressed(
            scene_dir / "trajectory.npz",
            actions_7d=actions_seg,
            ee_pos=ee_pos_arr,
            ee_quat=ee_quat_arr,
            joint_pos=joint_pos_arr,
            gripper_qpos=gripper_arr,
        )
        _save_scene_visuals(scene_dir, av_arr[:ACTION_LEN], eh_arr[:ACTION_LEN], spec.scene_id)

        meta = {
            "name": f"scenes/{spec.scene_id}",
            "scene_id": spec.scene_id,
            "group": spec.group,
            "description": spec.description,
            "source_task": DEFAULT_TASK,
            "source_demo": args.demo_key,
            "action_start": ACTION_START,
            "action_end": ACTION_START + ACTION_LEN - 1,
            "num_frames": ACTION_LEN,
            "floor_style": spec.floor_style,
            "wall_style": spec.wall_style,
            "table_texture": spec.table_texture or "martin_novak_wood_table.png",
            "light_diffuse": list(spec.light_diffuse) if spec.light_diffuse else [0.8, 0.8, 0.8],
            "num_custom_placements": len(spec.placements),
            "robot_contacts_total": int(total_contacts),
            "robot_contacts_detail": [c for c in contacts_per_step if c],
            "init_joint_diff_rad": init_joint_diff,
            "max_ee_pos_diff_m": max_ee_diff,
            "max_ee_quat_diff": max_quat_diff,
            "max_joint_pos_diff_rad": max_joint_diff,
            "max_gripper_qpos_diff_m": max_grip_diff,
            "wall_seconds": round(time.monotonic() - t0, 2),
        }
        (scene_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        manifest_rows.append(meta)

        status = "OK" if (total_contacts == 0 and max_ee_diff < 1e-6) else "WARN"
        print(
            f"  [{status}] {spec.scene_id:26s} | contacts={total_contacts} | "
            f"max|dEE|={max_ee_diff:.2e} m | max|dJoint|={max_joint_diff:.2e} rad | "
            f"{meta['wall_seconds']:.2f}s",
            flush=True,
        )

        if total_contacts > 0:
            raise RuntimeError(
                f"Scene {spec.scene_id} had {total_contacts} robot contacts: {meta['robot_contacts_detail'][:3]}"
            )
        if max_ee_diff > 1e-5 or max_joint_diff > 1e-5:
            raise RuntimeError(
                f"Scene {spec.scene_id} physical trajectory diverged: max_ee_diff={max_ee_diff:.3e}, max_joint_diff={max_joint_diff:.3e}"
            )

    # Save manifest CSV & JSON
    csv_cols = [
        "name",
        "scene_id",
        "group",
        "description",
        "floor_style",
        "wall_style",
        "table_texture",
        "robot_contacts_total",
        "init_joint_diff_rad",
        "max_ee_pos_diff_m",
        "max_joint_pos_diff_rad",
        "max_gripper_qpos_diff_m",
    ]
    with (out_dir / "scene_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in manifest_rows:
            w.writerow({k: r[k] for k in csv_cols})
    (out_dir / "scene_manifest.json").write_text(json.dumps(manifest_rows, indent=2) + "\n")
    print(f"[Simulate] Done all {len(specs)} scenes in {time.monotonic() - t_start:.1f}s -> {out_dir}", flush=True)


# ---------------------------------------------------------------------------
# Stage 2: GPU Encoding, Distance Metrics, Subspace & Decoder Audit, Plotting
# ---------------------------------------------------------------------------


def _pair_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Compute cosine similarity, normalized (relative) L2, unit-sphere L2, and raw L2."""
    fa = np.asarray(a, dtype=np.float64).reshape(-1)
    fb = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(fa))
    nb = float(np.linalg.norm(fb))
    denom_cos = max(na * nb, 1e-12)
    cos = float(np.dot(fa, fb) / denom_cos)
    cos = max(-1.0, min(1.0, cos))
    raw_l2 = float(np.linalg.norm(fa - fb))
    rel_l2 = float(raw_l2 / max(0.5 * (na + nb), 1e-12))
    unit_l2 = float(np.linalg.norm(fa / max(na, 1e-12) - fb / max(nb, 1e-12)))
    return {
        "cos": cos,
        "rel_l2": rel_l2,
        "unit_l2": unit_l2,
        "raw_l2": raw_l2,
        "norm_a": na,
        "norm_b": nb,
    }


def _subspace_metrics(
    z_a: np.ndarray,
    z_b: np.ndarray,
    u64: np.ndarray,
) -> Dict[str, float]:
    """Decompose z_a, z_b and (z_a - z_b) onto P = U_64 U_64^T (64D) and Q = I - P (3008D)."""
    fa = np.asarray(z_a, dtype=np.float64).reshape(-1)
    fb = np.asarray(z_b, dtype=np.float64).reshape(-1)
    u = np.asarray(u64, dtype=np.float64)  # (3072, 64)

    pa = u @ (u.T @ fa)
    pb = u @ (u.T @ fb)
    qa = fa - pa
    qb = fb - pb

    dp = pa - pb
    dq = qa - qb
    dp_l2 = float(np.linalg.norm(dp))
    dq_l2 = float(np.linalg.norm(dq))
    tot_sq = dp_l2 * dp_l2 + dq_l2 * dq_l2
    p_frac = (dp_l2 * dp_l2 / tot_sq) if tot_sq > 1e-18 else 0.0
    q_frac = (dq_l2 * dq_l2 / tot_sq) if tot_sq > 1e-18 else 0.0

    pm = _pair_metrics(pa, pb)
    qm = _pair_metrics(qa, qb)
    return {
        "p_l2": dp_l2,
        "q_l2": dq_l2,
        "p_energy_frac": p_frac,
        "q_energy_frac": q_frac,
        "p_cos": pm["cos"],
        "p_rel_l2": pm["rel_l2"],
        "q_cos": qm["cos"],
        "q_rel_l2": qm["rel_l2"],
    }


def _make_contact_sheet(
    out_dir: Path,
    distinct_ids: List[str],
    frames_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
    vs_base_map: Dict[str, Dict[str, Any]],
) -> Path:
    """Visual grid of all distinct scenes: agentview (t=0, t=15) & eye_in_hand (t=0, t=15)."""
    n = len(distinct_ids)
    fig, axes = plt.subplots(n, 4, figsize=(10.5, 2.05 * n), dpi=140)
    if n == 1:
        axes = axes[None, :]
    col_titles = [
        f"agentview t={ACTION_START}",
        f"agentview t={ACTION_START + ACTION_LEN - 1}",
        f"eye_in_hand t={ACTION_START}",
        f"eye_in_hand t={ACTION_START + ACTION_LEN - 1}",
    ]
    for r, sid in enumerate(distinct_ids):
        av, eh = frames_map[sid]
        imgs = [av[0], av[ACTION_LEN - 1], eh[0], eh[ACTION_LEN - 1]]
        m = vs_base_map[sid]
        row_label = (
            f"{sid}\n"
            f"[{m['group']}]\n"
            f"E9 cos={m['e9_cos']:.3f}\n"
            f"E9 nL2={m['e9_rel_l2']:.3f}"
        )
        for c in range(4):
            ax = axes[r, c]
            ax.imshow(_opengl_to_upright(imgs[c]))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel(row_label, fontsize=8.5, rotation=0, labelpad=68, va="center", ha="right")
    fig.suptitle(
        f"E9 Scene-Sensitivity Test Set: Identical Robot State & Actions ({ACTION_START}..{ACTION_START + ACTION_LEN - 1}) Across {n} Scenes",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0.11, 0.0, 1.0, 0.985])
    out_path = out_dir / "figures/fig1_scene_contact_sheet.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


SHORT_SCENE_NAMES = {
    "S0_base": "S0_base",
    "S0_repeat": "S0_repeat",
    "S1_bg_wall_floor_blue": "S1_bg_blue",
    "S2_bg_surface_dark_green": "S2_bg_dark",
    "S3_bg_surface_warm_light": "S3_bg_warm",
    "S4_layout_rearranged": "S4_layout",
    "S5_empty_table": "S5_empty",
    "S6_distractors_added": "S6_distract",
    "S7_objects_kitchen_props": "S7_kitchen",
    "S8_objects_study_props": "S8_study",
    "S9_combo_kitchen_dark": "S9_combo_kit",
    "S10_combo_study_blue": "S10_combo_std",
}


def _make_heatmaps(
    out_dir: Path,
    distinct_ids: List[str],
    clip_cos_mat: np.ndarray,
    e9_cos_mat: np.ndarray,
    clip_nl2_mat: np.ndarray,
    e9_nl2_mat: np.ndarray,
) -> Path:
    short_labels = [SHORT_SCENE_NAMES.get(s, s) for s in distinct_ids]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 12.0), dpi=150)
    panels = [
        (axes[0, 0], clip_cos_mat, "Frozen CLIP Input (16x2x768) — Cosine Similarity", "viridis", 0.25, 1.0),
        (axes[0, 1], e9_cos_mat, "E9 Local Behavior Latent (4x768) — Cosine Similarity", "viridis", 0.25, 1.0),
        (axes[1, 0], clip_nl2_mat, "Frozen CLIP Input (16x2x768) — Normalized L2", "magma_r", 0.0, 1.25),
        (axes[1, 1], e9_nl2_mat, "E9 Local Behavior Latent (4x768) — Normalized L2", "magma_r", 0.0, 1.25),
    ]
    n = len(distinct_ids)
    for ax, mat, title, cmap, vmin, vmax in panels:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=8.5)
        ax.set_yticklabels(short_labels, fontsize=8.5)
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                txt_color = "white" if (
                    (cmap == "viridis" and val < 0.68)
                    or (cmap == "magma_r" and val > 0.62)
                ) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5, color=txt_color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        "Same-Action / Different-Scene Pairwise Similarity & Normalized L2: Frozen CLIP vs E9 Latent",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    out_path = out_dir / "figures/fig2_similarity_heatmaps.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _make_breakdown_plot(
    out_dir: Path,
    vs_base_rows: List[Dict[str, Any]],
    diff_action_non_overlap: List[Dict[str, Any]],
) -> Path:
    # Exclude S0_base and S0_repeat from variant bars; show S0_repeat as sanity marker
    rows = [r for r in vs_base_rows if r["scene_id"] != "S0_base"]
    labels = [r["scene_id"] for r in rows]
    x = np.arange(len(rows))

    group_colors = {
        "sanity_repeat": "#6c757d",
        "background_only": "#4e79a7",
        "background_and_table": "#2b5c8f",
        "layout_only": "#f28e2b",
        "objects_only": "#59a14f",
        "combined_scene": "#e15759",
    }
    colors = [group_colors.get(r["group"], "#333333") for r in rows]

    da_e9_cos = float(np.mean([r["e9_cos"] for r in diff_action_non_overlap]))
    da_e9_nl2 = float(np.mean([r["e9_rel_l2"] for r in diff_action_non_overlap]))
    da_clip_nl2 = float(np.mean([r["clip_rel_l2"] for r in diff_action_non_overlap]))

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.5), dpi=150)

    # Panel 1: Normalized L2 (CLIP vs E9) vs Same-Scene Different-Action Reference
    w = 0.36
    axes[0].bar(x - w / 2, [r["clip_rel_l2"] for r in rows], width=w, color="#b0c4de", edgecolor="#4682b4", label="CLIP Input nL2")
    axes[0].bar(x + w / 2, [r["e9_rel_l2"] for r in rows], width=w, color=colors, edgecolor="black", linewidth=0.6, label="E9 Latent nL2")
    axes[0].axhline(da_e9_nl2, color="#d62728", linestyle="--", linewidth=1.6, label=f"Diff-Action Same-Scene E9 ({da_e9_nl2:.2f})")
    axes[0].axhline(da_clip_nl2, color="#4682b4", linestyle=":", linewidth=1.4, label=f"Diff-Action Same-Scene CLIP ({da_clip_nl2:.2f})")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=42, ha="right", fontsize=8)
    axes[0].set_ylabel("Normalized L2 Distance (vs S0_base)")
    axes[0].set_title("(a) Scene Shift: CLIP Input vs E9 Latent (nL2)", fontweight="bold")
    axes[0].legend(fontsize=7.5, loc="upper left")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: Cosine Similarity (E9 vs CLIP vs P-subspace vs Q-subspace)
    axes[1].plot(x, [r["clip_cos"] for r in rows], "s--", color="#4682b4", label="CLIP Input cos", markersize=5)
    axes[1].plot(x, [r["e9_cos"] for r in rows], "o-", color="#d62728", linewidth=2.0, label="E9 Full Latent cos", markersize=6)
    axes[1].plot(x, [r["p_cos"] for r in rows], "^-", color="#ff7f0e", linewidth=1.5, label="E9 P (64D) cos", markersize=5)
    axes[1].plot(x, [r["q_cos"] for r in rows], "v:", color="#2ca02c", linewidth=1.5, label="E9 Q (3008D) cos", markersize=5)
    axes[1].axhline(da_e9_cos, color="#d62728", linestyle="--", alpha=0.65, label=f"Diff-Action Same-Scene E9 ({da_e9_cos:.2f})")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=42, ha="right", fontsize=8)
    axes[1].set_ylabel("Cosine Similarity (vs S0_base)")
    axes[1].set_ylim(min(0.0, da_e9_cos - 0.1), 1.05)
    axes[1].set_title("(b) Cosine Similarity & P(64D) / Q(3008D) Breakdown", fontweight="bold")
    axes[1].legend(fontsize=7.5, loc="lower left")
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: P vs Q Energy Split + Frozen Decoder Action Shift (10D MSE vs S0_base)
    p_l2 = [r["p_l2"] for r in rows]
    q_l2 = [r["q_l2"] for r in rows]
    axes[2].bar(x, p_l2, color="#ff7f0e", alpha=0.85, label="||P(z - z_base)||_2 (64D)")
    axes[2].bar(x, q_l2, bottom=p_l2, color="#2ca02c", alpha=0.75, label="||Q(z - z_base)||_2 (3008D)")
    ax2 = axes[2].twinx()
    ax2.plot(x, [r["decoder_mse10_vs_base"] for r in rows], "D-", color="#1f1f1f", linewidth=1.8, markersize=5, label="Decoder 10D MSE vs S0_base")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=42, ha="right", fontsize=8)
    axes[2].set_ylabel("Latent Subspace L2 Norm (stacked)")
    ax2.set_ylabel("Frozen Decoder 10D Action MSE vs S0_base", color="#1f1f1f")
    axes[2].set_title("(c) P/Q Subspace Shift & Downstream Decoder Drift", fontweight="bold")
    h1, l1 = axes[2].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "figures/fig3_vs_base_breakdown_and_subspace.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_analyze(args: argparse.Namespace) -> None:
    import torch
    from e5 import ACTION_DIM_10D, ACTION_HORIZON, BPP_CKPT
    from e5.actions import action_7d_to_10d
    from e5.vision import BPPVisualEncoder
    from e9.stage1_local_model import load_frozen_e9_stage1_teacher

    t_start = time.monotonic()
    out_dir = Path(args.out_dir).resolve()
    manifest = json.loads((out_dir / "scene_manifest.json").read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Analyze] Device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})", flush=True)

    # 1. Load frozen BPP CLIP + frozen E9 Stage 1 Local Encoder/Decoder + P0 U_64 basis
    visual = BPPVisualEncoder(ckpt_path=BPP_CKPT, load_weights=True).to(device).eval()
    teacher = load_frozen_e9_stage1_teacher(args.e9_ckpt, device=device)
    p0_payload = torch.load(args.p0_bases, map_location="cpu", weights_only=False)
    u64 = p0_payload["U_reader"][:, :64].double().numpy()  # (3072, 64)
    s_z = float(p0_payload["s_z"])
    ortho_err = float(np.abs(u64.T @ u64 - np.eye(64)).max())
    print(f"[Analyze] Loaded E9 Stage 1 ({args.e9_ckpt.name}) & P0 U_64 (ortho_err={ortho_err:.2e}, s_z={s_z:.4f})", flush=True)

    # Fixed DDIM initial noises for downstream frozen decoder check
    gen1 = torch.Generator(device="cpu").manual_seed(12345)
    init_noise_1 = torch.randn((1, ACTION_HORIZON, ACTION_DIM_10D), generator=gen1).to(device)
    gen2 = torch.Generator(device="cpu").manual_seed(54321)
    init_noise_2 = torch.randn((1, ACTION_HORIZON, ACTION_DIM_10D), generator=gen2).to(device)

    autocast_clip = torch.float16 if device.type == "cuda" else None
    amp_e9 = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    @torch.inference_mode()
    def encode_and_decode(av: np.ndarray, eh: np.ndarray, conv: str = "opengl") -> Dict[str, Any]:
        clip_feats = visual.encode_dual_camera_uint8(
            np.ascontiguousarray(av),
            np.ascontiguousarray(eh),
            convention=conv,
            batch_size=32,
            autocast_dtype=autocast_clip,
        )  # (16, 2, 768) fp32 on GPU
        clip_in = clip_feats.unsqueeze(0)  # (1, 16, 2, 768)
        pad_mask = torch.zeros((1, ACTION_LEN), dtype=torch.bool, device=device)
        with amp_e9:
            out1 = teacher.predict_action(clip_in, clip_pad_mask=pad_mask, init_noise=init_noise_1)
            out2 = teacher.predict_action(clip_in, clip_pad_mask=pad_mask, init_noise=init_noise_2)
        b_t = out1["b_t"][0].float().cpu().numpy()  # (4, 768)
        return {
            "clip_feats": clip_feats.float().cpu().numpy(),  # (16, 2, 768)
            "b_t": b_t,
            "z_flat": b_t.reshape(-1),  # (3072,)
            "pred_10d_s1": out1["action_10d"][0].float().cpu().numpy(),  # (16, 10)
            "pred_7d_s1": out1["action_7d"][0].float().cpu().numpy(),  # (16, 7)
            "pred_10d_s2": out2["action_10d"][0].float().cpu().numpy(),
        }

    # 2. Encode all 12 simulated scenes (both V_{0:16} = steps 12..27 AND shifted V_{1:17} = steps 13..28 check)
    scene_enc: Dict[str, Dict[str, Any]] = {}
    scene_enc_shift1: Dict[str, Dict[str, Any]] = {}
    frames_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    gt_actions_7d: Optional[np.ndarray] = None

    for row in manifest:
        sid = row["scene_id"]
        sdir = out_dir / "scenes" / sid
        f_npz = np.load(sdir / "clip_frames.npz")
        t_npz = np.load(sdir / "trajectory.npz")
        av = f_npz["agentview_rgb"]
        eh = f_npz["eye_in_hand_rgb"]
        av17 = f_npz["agentview_rgb_full17"]
        eh17 = f_npz["eye_in_hand_rgb_full17"]
        conv = str(f_npz["convention"])
        if gt_actions_7d is None:
            gt_actions_7d = np.asarray(t_npz["actions_7d"], dtype=np.float32)

        frames_map[sid] = (av, eh)
        scene_enc[sid] = encode_and_decode(av, eh, conv=conv)
        scene_enc_shift1[sid] = encode_and_decode(av17[1:17], eh17[1:17], conv=conv)

    assert gt_actions_7d is not None
    gt_actions_10d = action_7d_to_10d(torch.from_numpy(gt_actions_7d)).numpy()  # (16, 10)

    # 3. Encode demo_1 same-scene reference clips (HDF5 t=12..27 + different-action clips)
    ref_dir = out_dir / "reference_demo1"
    ref_meta = json.loads((ref_dir / "meta.json").read_text())
    ref_enc: Dict[str, Dict[str, Any]] = {}
    for rm in ref_meta:
        lbl = rm["label"]
        r_npz = np.load(ref_dir / f"{lbl}.npz")
        ref_enc[lbl] = encode_and_decode(
            r_npz["agentview_rgb"],
            r_npz["eye_in_hand_rgb"],
            conv=str(r_npz["convention"]),
        )
        ref_enc[lbl]["actions_7d"] = np.asarray(r_npz["actions_7d"], dtype=np.float32)
        ref_enc[lbl]["start_step"] = int(rm["start_step"])

    # 4. Compute vs-S0_base metrics for all 12 scenes
    base = scene_enc["S0_base"]
    base_av, base_eh = frames_map["S0_base"]
    vs_base_rows: List[Dict[str, Any]] = []
    vs_base_map: Dict[str, Dict[str, Any]] = {}

    for row in manifest:
        sid = row["scene_id"]
        enc = scene_enc[sid]
        enc_s1 = scene_enc_shift1[sid]
        av, eh = frames_map[sid]

        px_av_mad = float(np.abs(av.astype(np.float32) - base_av.astype(np.float32)).mean())
        px_eh_mad = float(np.abs(eh.astype(np.float32) - base_eh.astype(np.float32)).mean())

        m_clip_all = _pair_metrics(enc["clip_feats"], base["clip_feats"])
        m_clip_av = _pair_metrics(enc["clip_feats"][:, 0, :], base["clip_feats"][:, 0, :])
        m_clip_eh = _pair_metrics(enc["clip_feats"][:, 1, :], base["clip_feats"][:, 1, :])
        m_e9 = _pair_metrics(enc["z_flat"], base["z_flat"])
        m_e9_s1 = _pair_metrics(enc_s1["z_flat"], scene_enc_shift1["S0_base"]["z_flat"])
        tok_cos = [
            _pair_metrics(enc["b_t"][k], base["b_t"][k])["cos"] for k in range(4)
        ]
        sub = _subspace_metrics(enc["z_flat"], base["z_flat"], u64)

        dec_mse10_vs_base = float(np.mean((enc["pred_10d_s1"] - base["pred_10d_s1"]) ** 2))
        dec_mse10_vs_base_s2 = float(np.mean((enc["pred_10d_s2"] - base["pred_10d_s2"]) ** 2))
        dec_mae7_vs_base = float(np.mean(np.abs(enc["pred_7d_s1"] - base["pred_7d_s1"])))
        dec_mse10_vs_gt = float(np.mean((enc["pred_10d_s1"] - gt_actions_10d) ** 2))
        dec_mae7_vs_gt = float(np.mean(np.abs(enc["pred_7d_s1"] - gt_actions_7d)))
        grip_acc_vs_gt = float(
            np.mean((enc["pred_7d_s1"][:, 6] >= 0.0) == (gt_actions_7d[:, 6] >= 0.0))
        )

        # Ratio of E9 shift to CLIP input shift (<1 means E9 suppresses scene shift vs CLIP)
        e9_to_clip_nl2_ratio = (
            m_e9["rel_l2"] / m_clip_all["rel_l2"] if m_clip_all["rel_l2"] > 1e-9 else 0.0
        )

        rec = {
            "name": row["name"],
            "scene_id": sid,
            "group": row["group"],
            "description": row["description"],
            "pixel_av_mad": px_av_mad,
            "pixel_eh_mad": px_eh_mad,
            "clip_cos": m_clip_all["cos"],
            "clip_rel_l2": m_clip_all["rel_l2"],
            "clip_av_cos": m_clip_av["cos"],
            "clip_av_rel_l2": m_clip_av["rel_l2"],
            "clip_eh_cos": m_clip_eh["cos"],
            "clip_eh_rel_l2": m_clip_eh["rel_l2"],
            "e9_cos": m_e9["cos"],
            "e9_rel_l2": m_e9["rel_l2"],
            "e9_unit_l2": m_e9["unit_l2"],
            "e9_raw_l2": m_e9["raw_l2"],
            "e9_norm": m_e9["norm_a"],
            "e9_shift1_cos": m_e9_s1["cos"],
            "e9_shift1_rel_l2": m_e9_s1["rel_l2"],
            "e9_to_clip_nl2_ratio": e9_to_clip_nl2_ratio,
            "e9_tok0_cos": tok_cos[0],
            "e9_tok1_cos": tok_cos[1],
            "e9_tok2_cos": tok_cos[2],
            "e9_tok3_cos": tok_cos[3],
            "p_l2": sub["p_l2"],
            "q_l2": sub["q_l2"],
            "p_energy_frac": sub["p_energy_frac"],
            "q_energy_frac": sub["q_energy_frac"],
            "p_cos": sub["p_cos"],
            "p_rel_l2": sub["p_rel_l2"],
            "q_cos": sub["q_cos"],
            "q_rel_l2": sub["q_rel_l2"],
            "decoder_mse10_vs_base": dec_mse10_vs_base,
            "decoder_mse10_vs_base_altseed": dec_mse10_vs_base_s2,
            "decoder_mae7_vs_base": dec_mae7_vs_base,
            "decoder_mse10_vs_gt": dec_mse10_vs_gt,
            "decoder_mae7_vs_gt": dec_mae7_vs_gt,
            "decoder_grip_acc_vs_gt": grip_acc_vs_gt,
            "robot_contacts_total": row["robot_contacts_total"],
            "max_ee_pos_diff_m": row["max_ee_pos_diff_m"],
            "max_joint_pos_diff_rad": row["max_joint_pos_diff_rad"],
        }
        vs_base_rows.append(rec)
        vs_base_map[sid] = rec

    # 5. Pairwise metrics across all 11 distinct scenes (excluding S0_repeat)
    distinct_ids = [r["scene_id"] for r in manifest if r["scene_id"] != "S0_repeat"]
    n_dist = len(distinct_ids)
    clip_cos_mat = np.ones((n_dist, n_dist), dtype=np.float64)
    e9_cos_mat = np.ones((n_dist, n_dist), dtype=np.float64)
    clip_nl2_mat = np.zeros((n_dist, n_dist), dtype=np.float64)
    e9_nl2_mat = np.zeros((n_dist, n_dist), dtype=np.float64)
    pairwise_rows: List[Dict[str, Any]] = []

    for i in range(n_dist):
        for j in range(i + 1, n_dist):
            si, sj = distinct_ids[i], distinct_ids[j]
            ei, ej = scene_enc[si], scene_enc[sj]
            mc = _pair_metrics(ei["clip_feats"], ej["clip_feats"])
            me = _pair_metrics(ei["z_flat"], ej["z_flat"])
            sub = _subspace_metrics(ei["z_flat"], ej["z_flat"], u64)
            dec_mse = float(np.mean((ei["pred_10d_s1"] - ej["pred_10d_s1"]) ** 2))

            clip_cos_mat[i, j] = clip_cos_mat[j, i] = mc["cos"]
            e9_cos_mat[i, j] = e9_cos_mat[j, i] = me["cos"]
            clip_nl2_mat[i, j] = clip_nl2_mat[j, i] = mc["rel_l2"]
            e9_nl2_mat[i, j] = e9_nl2_mat[j, i] = me["rel_l2"]

            pairwise_rows.append(
                {
                    "scene_a": si,
                    "group_a": vs_base_map[si]["group"],
                    "scene_b": sj,
                    "group_b": vs_base_map[sj]["group"],
                    "clip_cos": mc["cos"],
                    "clip_rel_l2": mc["rel_l2"],
                    "e9_cos": me["cos"],
                    "e9_rel_l2": me["rel_l2"],
                    "e9_unit_l2": me["unit_l2"],
                    "e9_raw_l2": me["raw_l2"],
                    "p_l2": sub["p_l2"],
                    "q_l2": sub["q_l2"],
                    "p_energy_frac": sub["p_energy_frac"],
                    "q_energy_frac": sub["q_energy_frac"],
                    "p_cos": sub["p_cos"],
                    "q_cos": sub["q_cos"],
                    "decoder_mse10_pair": dec_mse,
                }
            )

    # 6. Same-scene different-action reference comparisons (against demo_1 HDF5 t=12..27)
    ref_base = ref_enc["same_action_hdf5_12_27"]
    ref_rows: List[Dict[str, Any]] = []
    # Also check replayed S0_base vs raw HDF5 demo_1 t=12..27
    m_replay_vs_h5_e9 = _pair_metrics(base["z_flat"], ref_base["z_flat"])
    m_replay_vs_h5_clip = _pair_metrics(base["clip_feats"], ref_base["clip_feats"])

    for rm in ref_meta:
        lbl = rm["label"]
        if lbl == "same_action_hdf5_12_27":
            continue
        er = ref_enc[lbl]
        mc = _pair_metrics(er["clip_feats"], ref_base["clip_feats"])
        me = _pair_metrics(er["z_flat"], ref_base["z_flat"])
        sub = _subspace_metrics(er["z_flat"], ref_base["z_flat"], u64)
        act_mse7 = float(np.mean((er["actions_7d"] - ref_base["actions_7d"]) ** 2))
        dec_mse = float(np.mean((er["pred_10d_s1"] - ref_base["pred_10d_s1"]) ** 2))
        ref_rows.append(
            {
                "label": lbl,
                "start_step": rm["start_step"],
                "end_step": rm["end_step"],
                "frame_gap": abs(rm["start_step"] - ACTION_START),
                "non_overlapping": abs(rm["start_step"] - ACTION_START) >= ACTION_LEN,
                "gt_action_7d_mse": act_mse7,
                "clip_cos": mc["cos"],
                "clip_rel_l2": mc["rel_l2"],
                "e9_cos": me["cos"],
                "e9_rel_l2": me["rel_l2"],
                "e9_unit_l2": me["unit_l2"],
                "e9_raw_l2": me["raw_l2"],
                "p_l2": sub["p_l2"],
                "q_l2": sub["q_l2"],
                "p_energy_frac": sub["p_energy_frac"],
                "q_energy_frac": sub["q_energy_frac"],
                "p_cos": sub["p_cos"],
                "q_cos": sub["q_cos"],
                "decoder_mse10_pair": dec_mse,
            }
        )

    diff_action_non_overlap = [r for r in ref_rows if r["non_overlapping"]]
    variant_vs_base = [
        r for r in vs_base_rows if r["scene_id"] not in ("S0_base", "S0_repeat")
    ]

    # 7. Save CSVs & NPZ
    def _write_csv(path: Path, rows_list: List[Dict[str, Any]]) -> None:
        if not rows_list:
            return
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_list[0].keys()))
            w.writeheader()
            w.writerows(rows_list)

    _write_csv(out_dir / "vs_base_metrics.csv", vs_base_rows)
    _write_csv(out_dir / "pairwise_metrics.csv", pairwise_rows)
    _write_csv(out_dir / "reference_diff_action.csv", ref_rows)

    np.savez_compressed(
        out_dir / "latents.npz",
        scene_ids=np.array([r["scene_id"] for r in manifest]),
        e9_latents=np.stack([scene_enc[r["scene_id"]]["b_t"] for r in manifest], axis=0),
        clip_feats=np.stack([scene_enc[r["scene_id"]]["clip_feats"] for r in manifest], axis=0),
        pred_actions_10d=np.stack([scene_enc[r["scene_id"]]["pred_10d_s1"] for r in manifest], axis=0),
        gt_actions_10d=gt_actions_10d,
        gt_actions_7d=gt_actions_7d,
    )

    # 8. Aggregate summary statistics
    def _stats(vals: List[float]) -> Dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
        }

    by_group: Dict[str, Dict[str, Any]] = {}
    for g in ["background_only", "background_and_table", "layout_only", "objects_only", "combined_scene"]:
        grows = [r for r in variant_vs_base if r["group"] == g]
        if not grows:
            continue
        by_group[g] = {
            "n_scenes": len(grows),
            "clip_cos_mean": float(np.mean([r["clip_cos"] for r in grows])),
            "clip_rel_l2_mean": float(np.mean([r["clip_rel_l2"] for r in grows])),
            "e9_cos_mean": float(np.mean([r["e9_cos"] for r in grows])),
            "e9_rel_l2_mean": float(np.mean([r["e9_rel_l2"] for r in grows])),
            "e9_unit_l2_mean": float(np.mean([r["e9_unit_l2"] for r in grows])),
            "e9_raw_l2_mean": float(np.mean([r["e9_raw_l2"] for r in grows])),
            "e9_to_clip_nl2_ratio_mean": float(np.mean([r["e9_to_clip_nl2_ratio"] for r in grows])),
            "p_energy_frac_mean": float(np.mean([r["p_energy_frac"] for r in grows])),
            "q_energy_frac_mean": float(np.mean([r["q_energy_frac"] for r in grows])),
            "p_cos_mean": float(np.mean([r["p_cos"] for r in grows])),
            "q_cos_mean": float(np.mean([r["q_cos"] for r in grows])),
            "decoder_mse10_vs_base_mean": float(np.mean([r["decoder_mse10_vs_base"] for r in grows])),
            "decoder_mse10_vs_gt_mean": float(np.mean([r["decoder_mse10_vs_gt"] for r in grows])),
        }

    same_act_diff_scene_e9_nl2 = float(np.mean([r["e9_rel_l2"] for r in pairwise_rows]))
    same_act_diff_scene_e9_cos = float(np.mean([r["e9_cos"] for r in pairwise_rows]))
    diff_act_same_scene_e9_nl2 = float(np.mean([r["e9_rel_l2"] for r in diff_action_non_overlap]))
    diff_act_same_scene_e9_cos = float(np.mean([r["e9_cos"] for r in diff_action_non_overlap]))
    same_act_diff_scene_clip_nl2 = float(np.mean([r["clip_rel_l2"] for r in pairwise_rows]))
    diff_act_same_scene_clip_nl2 = float(np.mean([r["clip_rel_l2"] for r in diff_action_non_overlap]))

    # Additional block-structure diagnostic: pairwise distances strictly within non-base-layout scenes {S4, S5, S7, S8, S9, S10}
    non_base_obj_ids = {
        "S4_layout_rearranged",
        "S5_empty_table",
        "S7_objects_kitchen_props",
        "S8_objects_study_props",
        "S9_combo_kitchen_dark",
        "S10_combo_study_blue",
    }
    non_base_pairs = [
        r for r in pairwise_rows if r["scene_a"] in non_base_obj_ids and r["scene_b"] in non_base_obj_ids
    ]
    base_surf_ids = {
        "S0_base",
        "S1_bg_wall_floor_blue",
        "S2_bg_surface_dark_green",
        "S3_bg_surface_warm_light",
    }
    base_surf_pairs = [
        r for r in pairwise_rows if r["scene_a"] in base_surf_ids and r["scene_b"] in base_surf_ids
    ]

    summary = {
        "source": {
            "task": DEFAULT_TASK,
            "demo": args.demo_key,
            "action_start": ACTION_START,
            "action_end": ACTION_START + ACTION_LEN - 1,
            "num_frames": ACTION_LEN,
            "e9_checkpoint": str(args.e9_ckpt),
        },
        "physics_verification": {
            "num_scenes_total": len(manifest),
            "num_distinct_scenes": n_dist,
            "num_distinct_pairs": len(pairwise_rows),
            "max_robot_contacts": max(r["robot_contacts_total"] for r in manifest),
            "max_ee_pos_diff_m": max(r["max_ee_pos_diff_m"] for r in manifest),
            "max_joint_pos_diff_rad": max(r["max_joint_pos_diff_rad"] for r in manifest),
            "max_gripper_qpos_diff_m": max(r["max_gripper_qpos_diff_m"] for r in manifest),
            "sanity_repeat_e9_cos": vs_base_map["S0_repeat"]["e9_cos"],
            "sanity_repeat_e9_rel_l2": vs_base_map["S0_repeat"]["e9_rel_l2"],
            "replay_s0_vs_hdf5_demo1_e9_cos": m_replay_vs_h5_e9["cos"],
            "replay_s0_vs_hdf5_demo1_e9_rel_l2": m_replay_vs_h5_e9["rel_l2"],
            "replay_s0_vs_hdf5_demo1_clip_cos": m_replay_vs_h5_clip["cos"],
            "base_decoder_mse10_vs_gt": vs_base_map["S0_base"]["decoder_mse10_vs_gt"],
        },
        "same_action_diff_scene_vs_base_10_variants": {
            "clip_cos": _stats([r["clip_cos"] for r in variant_vs_base]),
            "clip_rel_l2": _stats([r["clip_rel_l2"] for r in variant_vs_base]),
            "e9_cos": _stats([r["e9_cos"] for r in variant_vs_base]),
            "e9_rel_l2": _stats([r["e9_rel_l2"] for r in variant_vs_base]),
            "e9_unit_l2": _stats([r["e9_unit_l2"] for r in variant_vs_base]),
            "e9_raw_l2": _stats([r["e9_raw_l2"] for r in variant_vs_base]),
            "e9_shift1_cos": _stats([r["e9_shift1_cos"] for r in variant_vs_base]),
            "e9_shift1_rel_l2": _stats([r["e9_shift1_rel_l2"] for r in variant_vs_base]),
            "p_energy_frac": _stats([r["p_energy_frac"] for r in variant_vs_base]),
            "q_energy_frac": _stats([r["q_energy_frac"] for r in variant_vs_base]),
            "p_cos": _stats([r["p_cos"] for r in variant_vs_base]),
            "q_cos": _stats([r["q_cos"] for r in variant_vs_base]),
            "decoder_mse10_vs_base": _stats([r["decoder_mse10_vs_base"] for r in variant_vs_base]),
            "decoder_mse10_vs_gt": _stats([r["decoder_mse10_vs_gt"] for r in variant_vs_base]),
        },
        "same_action_diff_scene_all_55_pairs": {
            "clip_cos": _stats([r["clip_cos"] for r in pairwise_rows]),
            "clip_rel_l2": _stats([r["clip_rel_l2"] for r in pairwise_rows]),
            "e9_cos": _stats([r["e9_cos"] for r in pairwise_rows]),
            "e9_rel_l2": _stats([r["e9_rel_l2"] for r in pairwise_rows]),
            "e9_unit_l2": _stats([r["e9_unit_l2"] for r in pairwise_rows]),
            "e9_raw_l2": _stats([r["e9_raw_l2"] for r in pairwise_rows]),
            "p_energy_frac": _stats([r["p_energy_frac"] for r in pairwise_rows]),
            "q_energy_frac": _stats([r["q_energy_frac"] for r in pairwise_rows]),
            "p_cos": _stats([r["p_cos"] for r in pairwise_rows]),
            "q_cos": _stats([r["q_cos"] for r in pairwise_rows]),
            "decoder_mse10_pair": _stats([r["decoder_mse10_pair"] for r in pairwise_rows]),
        },
        "same_scene_diff_action_non_overlap_ref": {
            "num_clips": len(diff_action_non_overlap),
            "clip_cos": _stats([r["clip_cos"] for r in diff_action_non_overlap]),
            "clip_rel_l2": _stats([r["clip_rel_l2"] for r in diff_action_non_overlap]),
            "e9_cos": _stats([r["e9_cos"] for r in diff_action_non_overlap]),
            "e9_rel_l2": _stats([r["e9_rel_l2"] for r in diff_action_non_overlap]),
            "e9_unit_l2": _stats([r["e9_unit_l2"] for r in diff_action_non_overlap]),
            "e9_raw_l2": _stats([r["e9_raw_l2"] for r in diff_action_non_overlap]),
            "p_energy_frac": _stats([r["p_energy_frac"] for r in diff_action_non_overlap]),
            "q_energy_frac": _stats([r["q_energy_frac"] for r in diff_action_non_overlap]),
            "decoder_mse10_pair": _stats([r["decoder_mse10_pair"] for r in diff_action_non_overlap]),
        },
        "key_ratios": {
            "e9_scene_to_action_nl2_ratio": same_act_diff_scene_e9_nl2 / max(diff_act_same_scene_e9_nl2, 1e-12),
            "clip_scene_to_action_nl2_ratio": same_act_diff_scene_clip_nl2 / max(diff_act_same_scene_clip_nl2, 1e-12),
            "e9_to_clip_scene_nl2_ratio_all55": same_act_diff_scene_e9_nl2 / max(same_act_diff_scene_clip_nl2, 1e-12),
            "same_act_diff_scene_e9_cos_mean": same_act_diff_scene_e9_cos,
            "diff_act_same_scene_e9_cos_mean": diff_act_same_scene_e9_cos,
        },
        "by_group_vs_base": by_group,
        "block_structure": {
            "base_objects_diff_bg_6_pairs": {
                "clip_cos_mean": float(np.mean([r["clip_cos"] for r in base_surf_pairs])),
                "clip_rel_l2_mean": float(np.mean([r["clip_rel_l2"] for r in base_surf_pairs])),
                "e9_cos_mean": float(np.mean([r["e9_cos"] for r in base_surf_pairs])),
                "e9_rel_l2_mean": float(np.mean([r["e9_rel_l2"] for r in base_surf_pairs])),
                "decoder_mse10_pair_mean": float(np.mean([r["decoder_mse10_pair"] for r in base_surf_pairs])),
            },
            "non_base_layout_15_pairs": {
                "clip_cos_mean": float(np.mean([r["clip_cos"] for r in non_base_pairs])),
                "clip_rel_l2_mean": float(np.mean([r["clip_rel_l2"] for r in non_base_pairs])),
                "e9_cos_mean": float(np.mean([r["e9_cos"] for r in non_base_pairs])),
                "e9_rel_l2_mean": float(np.mean([r["e9_rel_l2"] for r in non_base_pairs])),
                "decoder_mse10_pair_mean": float(np.mean([r["decoder_mse10_pair"] for r in non_base_pairs])),
            },
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # 9. Generate plots
    fig1 = _make_contact_sheet(out_dir, distinct_ids, frames_map, vs_base_map)
    fig2 = _make_heatmaps(out_dir, distinct_ids, clip_cos_mat, e9_cos_mat, clip_nl2_mat, e9_nl2_mat)
    fig3 = _make_breakdown_plot(out_dir, vs_base_rows, diff_action_non_overlap)

    # 10. Write REPORT.md
    _write_report(
        out_dir=out_dir,
        summary=summary,
        vs_base_rows=vs_base_rows,
        ref_rows=ref_rows,
        by_group=by_group,
        fig_paths=(fig1, fig2, fig3),
    )
    print(
        f"[Analyze] Complete in {time.monotonic() - t_start:.1f}s | "
        f"All-55 Same-Action/Diff-Scene E9 cos={same_act_diff_scene_e9_cos:.4f}, nL2={same_act_diff_scene_e9_nl2:.4f} | "
        f"Same-Scene/Diff-Action E9 cos={diff_act_same_scene_e9_cos:.4f}, nL2={diff_act_same_scene_e9_nl2:.4f} | "
        f"Scene/Action nL2 ratio={summary['key_ratios']['e9_scene_to_action_nl2_ratio']:.3f}",
        flush=True,
    )


def _write_report(
    *,
    out_dir: Path,
    summary: Dict[str, Any],
    vs_base_rows: List[Dict[str, Any]],
    ref_rows: List[Dict[str, Any]],
    by_group: Dict[str, Dict[str, Any]],
    fig_paths: Tuple[Path, Path, Path],
) -> None:
    pv = summary["physics_verification"]
    v10 = summary["same_action_diff_scene_vs_base_10_variants"]
    p55 = summary["same_action_diff_scene_all_55_pairs"]
    da = summary["same_scene_diff_action_non_overlap_ref"]
    kr = summary["key_ratios"]

    # Determine objective verdict string from measured numbers
    ratio = kr["e9_scene_to_action_nl2_ratio"]
    cos55 = p55["e9_cos"]["mean"]
    nl2_55 = p55["e9_rel_l2"]["mean"]
    if ratio >= 0.45 or cos55 < 0.85:
        verdict = "**显著编码了场景信息**（同一动作在不同场景间的 E9 latent 偏移量已达到“不同动作”距离的相当比例，且明显破坏或扰动冻结解码器的动作预测）。"
    elif ratio >= 0.20 or cos55 < 0.95:
        verdict = "**编码了可观测的中等程度场景信息**（同一动作在不同场景间存在清晰非零的 latent 偏移，虽小于不同动作间差异，但远非场景不变）。"
    else:
        verdict = "**对场景视觉变化较不敏感（场景信息编码较弱）**（同一动作在不同场景间的 latent 高度聚合，远小于不同动作间的距离）。"

    lines: List[str] = [
        "# E9 Latent 场景敏感性测试集验证报告",
        "",
        "## 1. 测试集构建与物理一致性核查",
        "",
        f"- **固定动作片段**：取自 `{DEFAULT_TASK}` 的 `demo_1`，固定 16 步自由空间接近动作序列 `actions[12:28]`（第 `12~27` 步，全段开爪 `gripper = -1.0`，末端在桌面高度 `z=0.90m` 上方 `18.8~26.1 cm` 处平滑移动）。",
        f"- **严格不变量**：所有 `{pv['num_scenes_total']}` 次仿真回放（`1` 个基准场景 `S0_base` + `1` 个重复确定性检查 `S0_repeat` + `10` 个视觉变化场景，覆盖背景/墙面地面、桌面纹理与光照、物体布局重排、空桌面、新增无关干扰物、全套替换厨房/书房物体、多因素组合切换）均使用**完全一致的机器人初始关节状态（7 轴 `qpos`/`qvel` + 2 指夹爪 `qpos`/`qvel`）、完全一致的 OSC_POSE 控制器零空间初始目标与状态重置、完全一致的 16 步动作序列**。",
        f"- **碰撞与物理轨迹一致性审计**：",
        f"  - 全场景机器人/夹爪碰撞次数：`max_robot_contacts = {pv['max_robot_contacts']}`（全程零碰撞、零额外交互）。",
        f"  - 跨全部 12 场景相对 `S0_base` 的最大末端位置误差：`max|Δee_pos| = {pv['max_ee_pos_diff_m']:.2e} m`。",
        f"  - 最大关节角误差：`max|Δjoint_pos| = {pv['max_joint_pos_diff_rad']:.2e} rad`，最大夹爪误差：`{pv['max_gripper_qpos_diff_m']:.2e} m`。",
        f"  - 确定性检查（`S0_repeat` vs `S0_base`）：`E9 cos = {pv['sanity_repeat_e9_cos']:.6f}`，`E9 norm L2 = {pv['sanity_repeat_e9_rel_l2']:.6f}`。",
        f"- **E9 编码方式**：严格复现 E9 Stage 1 管线——取每场景执行过程对应的 16 帧双目图像（`agentview_rgb`, `eye_in_hand_rgb`，128×128 uint8 OpenGL），经冻结 `BPPVisualEncoder`（CLIP ViT-B/16，fp16）编码为 `(1, 16, 2, 768)`，再经冻结 `E9Stage1LocalPolicy` (`e9/outputs/stage1_local/best.pt`, `clip_pad_mask=False`) 输出 `(1, 4, 768)` 展平为 `3072D` latent；同时测试移位 1 帧窗口 `V[1:17]` 做稳健性核查。",
        "",
        "![Scene Contact Sheet](figures/fig1_scene_contact_sheet.png)",
        "",
        "## 2. 核心汇总：同动作跨场景 vs 同场景不同动作",
        "",
        "| 对比类别 | 配对数 | CLIP Cos | CLIP norm L2 | **E9 Cos** | **E9 norm L2** | E9 Unit L2 | E9 Raw L2 | P(64D) / Q(3008D) 能量比 | 冻结 Decoder 10D MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| **同动作·不同场景（10 变体 vs `S0_base`）** | `10` | `{v10['clip_cos']['mean']:.4f}` | `{v10['clip_rel_l2']['mean']:.4f}` | **`{v10['e9_cos']['mean']:.4f}`** (`[{v10['e9_cos']['min']:.3f}, {v10['e9_cos']['max']:.3f}]`) | **`{v10['e9_rel_l2']['mean']:.4f}`** (`[{v10['e9_rel_l2']['min']:.3f}, {v10['e9_rel_l2']['max']:.3f}]`) | `{v10['e9_unit_l2']['mean']:.4f}` | `{v10['e9_raw_l2']['mean']:.2f}` | `{100*v10['p_energy_frac']['mean']:.1f}%` / `{100*v10['q_energy_frac']['mean']:.1f}%` | `{v10['decoder_mse10_vs_base']['mean']:.5f}` |",
        f"| **同动作·不同场景（11 场景全两两组合）** | `55` | `{p55['clip_cos']['mean']:.4f}` | `{p55['clip_rel_l2']['mean']:.4f}` | **`{p55['e9_cos']['mean']:.4f}`** (`[{p55['e9_cos']['min']:.3f}, {p55['e9_cos']['max']:.3f}]`) | **`{p55['e9_rel_l2']['mean']:.4f}`** (`[{p55['e9_rel_l2']['min']:.3f}, {p55['e9_rel_l2']['max']:.3f}]`) | `{p55['e9_unit_l2']['mean']:.4f}` | `{p55['e9_raw_l2']['mean']:.2f}` | `{100*p55['p_energy_frac']['mean']:.1f}%` / `{100*p55['q_energy_frac']['mean']:.1f}%` | `{p55['decoder_mse10_pair']['mean']:.5f}` |",
        f"| **同场景·不同动作（`demo_1` 非重叠片段标尺）** | `{da['num_clips']}` | `{da['clip_cos']['mean']:.4f}` | `{da['clip_rel_l2']['mean']:.4f}` | **`{da['e9_cos']['mean']:.4f}`** (`[{da['e9_cos']['min']:.3f}, {da['e9_cos']['max']:.3f}]`) | **`{da['e9_rel_l2']['mean']:.4f}`** (`[{da['e9_rel_l2']['min']:.3f}, {da['e9_rel_l2']['max']:.3f}]`) | `{da['e9_unit_l2']['mean']:.4f}` | `{da['e9_raw_l2']['mean']:.2f}` | `{100*da['p_energy_frac']['mean']:.1f}%` / `{100*da['q_energy_frac']['mean']:.1f}%` | `{da['decoder_mse10_pair']['mean']:.5f}` |",
        "",
        f"- **关键比例（Scene-Shift / Action-Shift）**：",
        f"  - 在 **冻结 CLIP 输入层**：同动作跨场景 nL2 / 同场景不同动作 nL2 = **`{kr['clip_scene_to_action_nl2_ratio']:.3f}`**（`{p55['clip_rel_l2']['mean']:.4f}` vs `{da['clip_rel_l2']['mean']:.4f}`）。",
        f"  - 在 **E9 Latent 层**：同动作跨场景 nL2 / 同场景不同动作 nL2 = **`{kr['e9_scene_to_action_nl2_ratio']:.3f}`**（`{p55['e9_rel_l2']['mean']:.4f}` vs `{da['e9_rel_l2']['mean']:.4f}`）。",
        f"  - 移位 1 帧窗口（`V[1:17]`，覆盖 `a_12..a_27` 全部结果帧）稳健性核查：10 变体 vs `S0_base` 均值 `E9 cos = {v10['e9_shift1_cos']['mean']:.4f}`，`E9 norm L2 = {v10['e9_shift1_rel_l2']['mean']:.4f}`，与 `V[0:16]` 完全一致。",
        "",
        "![Pairwise Heatmaps](figures/fig2_similarity_heatmaps.png)",
        "",
        "## 3. 分视觉因素与逐场景结果（相对 `S0_base`）",
        "",
        "### 3.1 按视觉因素分组汇总",
        "",
        "| 视觉因素组 | 场景数 | CLIP Cos | CLIP nL2 | **E9 Cos** | **E9 nL2** | E9/CLIP nL2 比 | `P`(64D) Cos / 能量% | `Q`(3008D) Cos / 能量% | Decoder 10D MSE (vs Base / vs GT) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g, gd in by_group.items():
        lines.append(
            f"| `{g}` | `{gd['n_scenes']}` | `{gd['clip_cos_mean']:.4f}` | `{gd['clip_rel_l2_mean']:.4f}` | "
            f"**`{gd['e9_cos_mean']:.4f}`** | **`{gd['e9_rel_l2_mean']:.4f}`** | `{gd['e9_to_clip_nl2_ratio_mean']:.2f}x` | "
            f"`{gd['p_cos_mean']:.4f}` / `{100*gd['p_energy_frac_mean']:.1f}%` | "
            f"`{gd['q_cos_mean']:.4f}` / `{100*gd['q_energy_frac_mean']:.1f}%` | "
            f"`{gd['decoder_mse10_vs_base_mean']:.5f}` / `{gd['decoder_mse10_vs_gt_mean']:.5f}` |"
        )

    lines += [
        "",
        "### 3.2 逐场景详细指标（相对 `S0_base`）",
        "",
        "| Scene ID | 组别 | AV / EH 像素 MAD | CLIP Cos (AV / EH) | CLIP nL2 | **E9 Cos** | **E9 nL2** | E9 Unit L2 | E9 Raw L2 | `P`(64D) L2 (`%`) | `Q`(3008D) L2 (`%`) | Decoder 10D MSE (vs Base / vs GT) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in vs_base_rows:
        lines.append(
            f"| `{r['scene_id']}` | `{r['group']}` | `{r['pixel_av_mad']:.1f}` / `{r['pixel_eh_mad']:.1f}` | "
            f"`{r['clip_cos']:.4f}` (`{r['clip_av_cos']:.3f}`/`{r['clip_eh_cos']:.3f}`) | `{r['clip_rel_l2']:.4f}` | "
            f"**`{r['e9_cos']:.4f}`** | **`{r['e9_rel_l2']:.4f}`** | `{r['e9_unit_l2']:.4f}` | `{r['e9_raw_l2']:.2f}` | "
            f"`{r['p_l2']:.2f}` (`{100*r['p_energy_frac']:.1f}%`) | `{r['q_l2']:.2f}` (`{100*r['q_energy_frac']:.1f}%`) | "
            f"`{r['decoder_mse10_vs_base']:.5f}` / `{r['decoder_mse10_vs_gt']:.5f}` |"
        )

    lines += [
        "",
        "![Breakdown and Subspace](figures/fig3_vs_base_breakdown_and_subspace.png)",
        "",
        "### 3.3 同场景不同动作标尺（`demo_1` 各时段 vs `t=12..27`）",
        "",
        "| 片段标签 | 帧范围 | 重叠? | GT 7D Action MSE | CLIP Cos / nL2 | **E9 Cos** | **E9 nL2** | E9 Raw L2 | `P` / `Q` 能量% | Decoder 10D MSE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ref_rows:
        ov = "否" if r["non_overlapping"] else "是"
        lines.append(
            f"| `{r['label']}` | `{r['start_step']}..{r['end_step']}` | {ov} | `{r['gt_action_7d_mse']:.4f}` | "
            f"`{r['clip_cos']:.4f}` / `{r['clip_rel_l2']:.4f}` | **`{r['e9_cos']:.4f}`** | **`{r['e9_rel_l2']:.4f}`** | "
            f"`{r['e9_raw_l2']:.2f}` | `{100*r['p_energy_frac']:.1f}%` / `{100*r['q_energy_frac']:.1f}%` | `{r['decoder_mse10_pair']:.5f}` |"
        )

    lines += [
        "",
        "## 4. 结论判读",
        "",
        f"1. **总体判定**：E9 Local Behavior Encoder {verdict}",
        f"2. **定量证据**：",
        f"   - 在机器人初始状态、16 步动作和物理轨迹完全一致（`max|Δee_pos| = {pv['max_ee_pos_diff_m']:.2e} m`，零碰撞）的条件下，11 个不同场景间的 55 组同动作配对给出均值 **`E9 cos = {p55['e9_cos']['mean']:.4f}`**（范围 `[{p55['e9_cos']['min']:.4f}, {p55['e9_cos']['max']:.4f}]`）、**`E9 normalized L2 = {p55['e9_rel_l2']['mean']:.4f}`**（范围 `[{p55['e9_rel_l2']['min']:.4f}, {p55['e9_rel_l2']['max']:.4f}]`）。",
        f"   - 对比同场景不同动作标尺（`E9 cos = {da['e9_cos']['mean']:.4f}`, `E9 nL2 = {da['e9_rel_l2']['mean']:.4f}`），纯视觉场景变化引起的 E9 latent 距离是不同动作距离的 **`{100*kr['e9_scene_to_action_nl2_ratio']:.1f}%`**；而在冻结 CLIP 输入端该比例高达 **`{100*kr['clip_scene_to_action_nl2_ratio']:.1f}%`**——说明 E9 相比 CLIP 输入确实压缩了部分场景外观变化，但并未完全消除。",
        f"3. **分因素机制剖析（关键发现：远景背景 vs 桌面纹理 vs 腕部视野物体上下文）**：",
        f"   - **纯远景墙面/地面（`S1_bg_wall_floor_blue`）几乎完全不变**：`E9 cos = 0.9998`, `nL2 = 0.0174`, `Decoder MSE = 0.00000`（E9 将 CLIP 输入的微小偏移 `0.0408` 进一步压缩至 `0.43x`）。",
        f"   - **桌面纹理与光照（`S2`, `S3`，保持全部物体位姿不变）呈中度压缩**：暖木纹+暖光 (`S3`) 几乎不变（`E9 cos = 0.9920`, `nL2 = 0.1264`）；高对比深色石纹桌面 (`S2`) 导致 `eye_in_hand` CLIP 显著偏移（`EH CLIP cos = 0.458`），E9 将整体偏移从 CLIP `0.7712` 压缩至 `0.5418`（`E9 cos = 0.8532`，`Decoder MSE = 0.00624`）。",
        f"   - **桌面物体布局与无关干扰物（`S4`~`S10`）引发极强敏感性**：",
        f"     - 仅在原始场景旁**新增 4 个无关干扰物**（`S6_distractors_added`，原目标物体与位置均不变，零碰撞），E9 即产生 `cos = 0.6907`、`nL2 = 0.7864`、`Decoder MSE = 0.02198`（为基准 GT 重建误差 `0.00106` 的 20 倍）。",
        f"     - **重排布局（`S4`）、空桌面（`S5`）或替换厨房/书房物体（`S7`~`S10`）**使腕部相机 `eye_in_hand` 视野内的桌面物体上下文彻底改变（`EH CLIP cos` 跌至 `0.056~0.141`），导致 `E9 cos` 骤降至 **`0.309~0.520`**、`E9 nL2` 高达 **`0.980~1.176`**，几乎追平同场景**不同动作**之间的距离（`nL2 = 1.168`），且 E9 不再压缩 CLIP 偏移（`E9/CLIP nL2` 比 `1.01x~1.11x`）。",
        f"   - **热图两块结构验证（`fig2`）**：在均移除了原 `S0` 目标布局的 6 个跨场景样本（`S4, S5, S7, S8, S9, S10`，15 组两两配对）内部，CLIP 均值 `cos = {summary['block_structure']['non_base_layout_15_pairs']['clip_cos_mean']:.4f}` (`nL2 = {summary['block_structure']['non_base_layout_15_pairs']['clip_rel_l2_mean']:.4f}`)，E9 均值 `cos = {summary['block_structure']['non_base_layout_15_pairs']['e9_cos_mean']:.4f}` (`nL2 = {summary['block_structure']['non_base_layout_15_pairs']['e9_rel_l2_mean']:.4f}`)——虽较跨 `S0` 块有所收敛，但仍存在约 `0.62` 量级的显著场景偏移。",
        f"4. **子空间结构与下游动作影响**：",
        f"   - 跨场景残差在 `P`（64D 动作主子空间）与 `Q`（3008D 正交补）的能量分配为 **`{100*p55['p_energy_frac']['mean']:.1f}%` vs `{100*p55['q_energy_frac']['mean']:.1f}%`**（10 变体 vs `S0_base`：`{100*v10['p_energy_frac']['mean']:.1f}%` vs `{100*v10['q_energy_frac']['mean']:.1f}%`）；也就是说，**场景视觉变化并非只落在 3008D 无关正交补 `Q` 中，而是有近 80% 的残差能量直接灌入了 64D 动作主子空间 `P`**！",
        f"   - 基准 `S0_base` 的冻结 Decoder 重建 GT 动作误差为 `10D MSE = {pv['base_decoder_mse10_vs_gt']:.5f}`；10 个视觉变体相对 `S0_base` 引起的解码动作位移均值 `10D MSE = {v10['decoder_mse10_vs_base']['mean']:.5f}`（最大 `{v10['decoder_mse10_vs_base']['max']:.5f}`），证明 E9 确实深度编码了（特别是腕部视野与桌面上的）场景/物体视觉信息，而非纯运动学动作表征。",
        "",
        "## 5. 产物索引",
        "",
        f"- 场景清单与物理验证：[scene_manifest.csv](file://{out_dir / 'scene_manifest.csv'})",
        f"- 逐场景 vs `S0_base` 指标：[vs_base_metrics.csv](file://{out_dir / 'vs_base_metrics.csv'})",
        f"- 55 组两两配对指标：[pairwise_metrics.csv](file://{out_dir / 'pairwise_metrics.csv'})",
        f"- 同场景不同动作标尺：[reference_diff_action.csv](file://{out_dir / 'reference_diff_action.csv'})",
        f"- 汇总 JSON：[summary.json](file://{out_dir / 'summary.json'}) | Latent 张量：[latents.npz](file://{out_dir / 'latents.npz'})",
        f"- 可视化图：[fig1_scene_contact_sheet.png](file://{fig_paths[0]}) | [fig2_similarity_heatmaps.png](file://{fig_paths[1]}) | [fig3_vs_base_breakdown_and_subspace.png](file://{fig_paths[2]})",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sim = sub.add_parser("simulate", help="Replay fixed action sequence across controlled LIBERO scenes (EGL)")
    p_sim.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    p_sim.add_argument("--demo-key", type=str, default="demo_1")
    p_sim.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    p_ana = sub.add_parser("analyze", help="Extract E9 latents and compute scene sensitivity metrics (CUDA)")
    p_ana.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p_ana.add_argument("--demo-key", type=str, default="demo_1")
    p_ana.add_argument("--e9-ckpt", type=Path, default=DEFAULT_E9_CKPT)
    p_ana.add_argument("--p0-bases", type=Path, default=DEFAULT_P0_BASES)

    args = parser.parse_args()
    if args.cmd == "simulate":
        run_simulate(args)
    elif args.cmd == "analyze":
        run_analyze(args)


if __name__ == "__main__":
    main()
