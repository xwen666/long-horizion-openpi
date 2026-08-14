import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import sys
from typing import Literal

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
LIBERO_PACKAGE_ROOTS = {
    "standard": pathlib.Path("/cc/openpi_wam/third_party/libero"),
    "plus": pathlib.Path("/cc/openpi_wam/LIBERO-plus"),
}
LIBERO_BENCHMARK_ROOTS = {
    "standard": pathlib.Path("/cc/openpi_wam/third_party/libero/libero/libero"),
    "plus": pathlib.Path("/cc/openpi_wam/LIBERO-plus/libero/libero"),
}


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 16

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    libero_mode: Literal["standard", "plus"] = "standard"
    task_start: int = 0
    max_tasks: int | None = None
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_videos: bool = True
    results_path: str | None = None
    abort_on_error: bool = False

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    benchmark, get_libero_path, OffScreenRenderEnv = _load_libero(args.libero_mode)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.save_videos:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    results_path = pathlib.Path(args.results_path) if args.results_path is not None else None
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        if results_path.exists():
            results_path.unlink()

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_end = num_tasks_in_suite if args.max_tasks is None else min(num_tasks_in_suite, args.task_start + args.max_tasks)
    task_ids = range(args.task_start, task_end)
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, get_libero_path, OffScreenRenderEnv)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            done = False
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    if args.save_videos:
                        replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute a new chunk.
                        # For WAM/Cosmos policies, the server calls WAM once inside this
                        # policy inference, then we execute the returned chunk open-loop.
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            # Keep Cosmos input orientation separate from the
                            # 180-degree-rotated images used by the OpenPI VLA.
                            "observation/cosmos_image": np.ascontiguousarray(obs["agentview_image"][::-1]),
                            "observation/cosmos_wrist_image": np.ascontiguousarray(
                                obs["robot0_eye_in_hand_image"][::-1]
                            ),
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.exception("Caught exception")
                    if args.abort_on_error:
                        raise
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            if args.save_videos:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            if results_path is not None:
                _append_jsonl(
                    results_path,
                    {
                        "type": "episode",
                        "libero_mode": args.libero_mode,
                        "task_suite_name": args.task_suite_name,
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "task": task_description,
                        "success": bool(done),
                        "total_episodes_so_far": total_episodes,
                        "total_successes_so_far": total_successes,
                    },
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        if results_path is not None:
            _append_jsonl(
                results_path,
                {
                    "type": "task_summary",
                    "libero_mode": args.libero_mode,
                    "task_suite_name": args.task_suite_name,
                    "task_id": task_id,
                    "task": task_description,
                    "episodes": task_episodes,
                    "successes": task_successes,
                    "success_rate": float(task_successes) / float(task_episodes),
                    "total_episodes_so_far": total_episodes,
                    "total_successes_so_far": total_successes,
                    "total_success_rate_so_far": float(total_successes) / float(total_episodes),
                },
            )

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    if results_path is not None:
        _append_jsonl(
            results_path,
            {
                "type": "suite_summary",
                "libero_mode": args.libero_mode,
                "task_suite_name": args.task_suite_name,
                "episodes": total_episodes,
                "successes": total_successes,
                "success_rate": float(total_successes) / float(total_episodes),
            },
        )


def _append_jsonl(path: pathlib.Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _write_libero_config(mode: str) -> None:
    benchmark_root = LIBERO_BENCHMARK_ROOTS[mode]
    config_dir = pathlib.Path("/tmp/openpi_libero_configs") / mode
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": benchmark_root.parent / "datasets",
        "assets": benchmark_root / "assets",
    }
    with (config_dir / "config.yaml").open("w", encoding="utf-8") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


def _load_libero(mode: str):
    package_root = LIBERO_PACKAGE_ROOTS[mode]
    _write_libero_config(mode)

    for module_name in list(sys.modules):
        if module_name == "libero" or module_name.startswith("libero."):
            del sys.modules[module_name]

    for root in LIBERO_PACKAGE_ROOTS.values():
        root_str = str(root)
        sys.path[:] = [path for path in sys.path if path != root_str]
    sys.path.insert(0, str(package_root))

    from libero.libero import benchmark  # noqa: PLC0415
    from libero.libero import get_libero_path  # noqa: PLC0415
    from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

    logging.info("Using %s LIBERO package from %s", mode, package_root)
    logging.info("Using LIBERO_CONFIG_PATH=%s", os.environ["LIBERO_CONFIG_PATH"])
    return benchmark, get_libero_path, OffScreenRenderEnv


def _get_libero_env(task, resolution, seed, get_libero_path, OffScreenRenderEnv):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
