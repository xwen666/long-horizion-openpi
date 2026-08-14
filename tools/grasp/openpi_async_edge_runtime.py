from __future__ import annotations

import argparse
import importlib.util
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi_client import websocket_client_policy


DATASET_ROOT = Path("/cc/openpi/grasp_200")
SPLITS_PATH = Path("/cc/openpi/grasp_200lora_splits/splits.json")


def load_piper_controller_class():
    spec = importlib.util.spec_from_file_location(
        "piper_interface_v2", Path("/cc/starVLA/piper_sdk/piper_sdk/interface/piper_interface_v2.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.C_PiperInterface_V2


class USBCamera:
    def __init__(self, index_or_path=0):
        if isinstance(index_or_path, str) and not index_or_path.isdigit():
            self.cap = cv2.VideoCapture(index_or_path, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(int(index_or_path))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        for _ in range(10):
            self.cap.read()

    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera read failed")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


class PiperController:
    def __init__(self, can_name: str = "can0"):
        import sys

        sys.path.insert(0, "/cc/starVLA/piper_sdk")
        from piper_sdk import C_PiperInterface_V2

        self.pi = C_PiperInterface_V2(can_name)
        self.pi.ConnectPort()
        while not self.pi.EnablePiper():
            time.sleep(0.01)

    def get_state(self) -> list[float]:
        joint_msgs = self.pi.GetArmJointMsgs()
        gripper_msgs = self.pi.GetArmGripperMsgs()
        return [
            joint_msgs.joint_state.joint_1 / 1000.0,
            joint_msgs.joint_state.joint_2 / 1000.0,
            joint_msgs.joint_state.joint_3 / 1000.0,
            joint_msgs.joint_state.joint_4 / 1000.0,
            joint_msgs.joint_state.joint_5 / 1000.0,
            joint_msgs.joint_state.joint_6 / 1000.0,
            gripper_msgs.gripper_state.grippers_angle / 1000.0,
        ]

    def execute_action(self, action: np.ndarray, action_threshold: float, speed_pct: int):
        joints = action[:6].tolist()
        gripper = float(1.0 if action[6] > action_threshold else 0.0)
        self.pi.MotionCtrl_2(0x01, 0x01, speed_pct, 0x00)
        self.pi.JointCtrl(
            round(joints[0] * 1000),
            round(joints[1] * 1000),
            round(joints[2] * 1000),
            round(joints[3] * 1000),
            round(joints[4] * 1000),
            round(joints[5] * 1000),
        )
        self.pi.GripperCtrl(round(gripper * 1000), 1000, 0x01, 0)

    def disconnect(self):
        self.pi.DisableArm()
        self.pi.DisconnectPort()


@dataclass
class InferenceTask:
    obs: dict
    tag: str


class AsyncBroker:
    def __init__(self, policy_client):
        self.policy_client = policy_client
        self._tasks: queue.Queue[InferenceTask] = queue.Queue(maxsize=1)
        self._latest = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, obs: dict, tag: str):
        while True:
            try:
                self._tasks.get_nowait()
            except queue.Empty:
                break
        self._tasks.put(InferenceTask(obs=obs, tag=tag))

    def _loop(self):
        while self._running:
            try:
                task = self._tasks.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            try:
                result = self.policy_client.infer(task.obs)
                payload = {
                    "ok": True,
                    "tag": task.tag,
                    "actions": np.asarray(result["actions"], dtype=np.float32),
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                }
            except Exception as exc:
                payload = {
                    "ok": False,
                    "tag": task.tag,
                    "error": str(exc),
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                }
            with self._lock:
                self._latest = payload

    def latest(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False


def load_episode_obs(episode_id: int):
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    metadata = lerobot_dataset.LeRobotDatasetMetadata("grasp_200lora", root=DATASET_ROOT)
    horizon = 20
    delta_timestamps = {"action": [t / metadata.fps for t in range(horizon)]}
    dataset = lerobot_dataset.LeRobotDataset(
        "grasp_200lora",
        root=DATASET_ROOT,
        episodes=None,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    episode_indices = torch.stack(dataset.hf_dataset["episode_index"]).numpy()
    matching = np.where(episode_indices == episode_id)[0]
    return dataset, matching


def make_obs(front_img, wrist_img, state, prompt):
    return {
        "observation/image": np.array(front_img, copy=True),
        "observation/wrist_image": np.array(wrist_img, copy=True),
        "observation/state": np.array(state, dtype=np.float32, copy=True),
        "actions": np.zeros((20, 7), dtype=np.float32, order="C"),
        "prompt": prompt,
    }


def run_episode_replay(policy_client, episode_id: int, action_horizon: int, action_threshold: float, output_path: Path):
    dataset, matching = load_episode_obs(episode_id)
    if len(matching) == 0:
        raise ValueError(f"Episode {episode_id} not found")

    broker = AsyncBroker(policy_client)
    period = 1.0 / 15.0
    replan_after = action_horizon // 2
    frames = []
    executed = []
    pred_chunk = None
    pred_idx = 0
    next_tag = 0
    next_requested = False

    try:
        for i, dataset_idx in enumerate(matching):
            sample = dataset[int(dataset_idx)]
            front_img = sample["observation.images.front"].permute(1, 2, 0).numpy()
            wrist_img = sample["observation.images.wrist"].permute(1, 2, 0).numpy()
            state = sample["observation.state"].numpy()
            prompt = sample["task"]

            obs = make_obs(front_img, wrist_img, state, prompt)
            if pred_chunk is None or pred_idx >= action_horizon:
                tag = f"chunk-{next_tag}"
                next_tag += 1
                broker.submit(obs, tag)
                while True:
                    result = broker.latest()
                    if result and result["tag"] == tag:
                        if not result["ok"]:
                            raise RuntimeError(result["error"])
                        pred_chunk = result["actions"]
                        pred_idx = 0
                        next_requested = False
                        break
                    time.sleep(0.005)

            action = pred_chunk[pred_idx].copy()
            action[-1] = 1.0 if action[-1] > action_threshold else 0.0
            executed.append(action.tolist())
            pred_idx += 1

            vis = np.concatenate([front_img, wrist_img], axis=1)
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.putText(vis, f"Episode {episode_id} frame {i}/{len(matching)-1}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, f"Pred action: {np.round(action, 3).tolist()}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            frames.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

            if pred_idx == replan_after and not next_requested:
                tag = f"chunk-{next_tag}"
                next_tag += 1
                broker.submit(obs, tag)
                next_requested = True

            if pred_idx >= action_horizon and next_requested:
                while True:
                    result = broker.latest()
                    if result and result["tag"] == f"chunk-{next_tag - 1}":
                        if not result["ok"]:
                            raise RuntimeError(result["error"])
                        pred_chunk = result["actions"]
                        pred_idx = 0
                        next_requested = False
                        break
                    time.sleep(0.005)

            time.sleep(period)
    finally:
        broker.stop()

    import imageio.v2 as imageio

    imageio.mimsave(output_path, frames, fps=15)
    return executed


def main():
    parser = argparse.ArgumentParser(description="Async edge runtime for openpi grasp policy")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--front-id", default="0")
    parser.add_argument("--wrist-id", default="2")
    parser.add_argument("--task", default="grasp the object")
    parser.add_argument("--action-horizon", type=int, default=20)
    parser.add_argument("--action-threshold", type=float, default=0.5)
    parser.add_argument("--speed", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replay-episode", type=int, default=None)
    parser.add_argument("--replay-output", default="/cc/openpi/outputs/openpi_async_replay.mp4")
    parser.add_argument("--can", default="can0")
    args = parser.parse_args()

    policy_client = websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)

    if args.replay_episode is not None:
        executed = run_episode_replay(
            policy_client,
            episode_id=args.replay_episode,
            action_horizon=args.action_horizon,
            action_threshold=args.action_threshold,
            output_path=Path(args.replay_output),
        )
        print(f"Saved replay video to {args.replay_output}")
        print(f"Executed {len(executed)} replay steps")
        return

    front_cam = USBCamera(args.front_id)
    wrist_cam = USBCamera(args.wrist_id)
    robot = None if args.dry_run else PiperController(can_name=args.can)
    broker = AsyncBroker(policy_client)
    period = 1.0 / 15.0
    replan_after = args.action_horizon // 2
    pred_chunk = None
    pred_idx = 0
    next_tag = 0
    next_requested = False

    try:
        while True:
            t0 = time.perf_counter()
            front_img = front_cam.read()
            wrist_img = wrist_cam.read()
            state = [0.0] * 7 if robot is None else robot.get_state()
            obs = make_obs(front_img, wrist_img, state, args.task)

            if pred_chunk is None or pred_idx >= args.action_horizon:
                tag = f"chunk-{next_tag}"
                next_tag += 1
                broker.submit(obs, tag)
                while True:
                    result = broker.latest()
                    if result and result["tag"] == tag:
                        if not result["ok"]:
                            raise RuntimeError(result["error"])
                        pred_chunk = result["actions"]
                        pred_idx = 0
                        next_requested = False
                        break
                    time.sleep(0.005)

            action = pred_chunk[pred_idx].copy()
            action[-1] = 1.0 if action[-1] > args.action_threshold else 0.0
            if robot is not None:
                robot.execute_action(action, args.action_threshold, args.speed)
            print(f"exec step {pred_idx}/{args.action_horizon}: {np.round(action, 4).tolist()}")
            pred_idx += 1

            if pred_idx == replan_after and not next_requested:
                tag = f"chunk-{next_tag}"
                next_tag += 1
                broker.submit(obs, tag)
                next_requested = True

            if pred_idx >= args.action_horizon and next_requested:
                while True:
                    result = broker.latest()
                    if result and result["tag"] == f"chunk-{next_tag - 1}":
                        if not result["ok"]:
                            raise RuntimeError(result["error"])
                        pred_chunk = result["actions"]
                        pred_idx = 0
                        next_requested = False
                        break
                    time.sleep(0.005)

            elapsed = time.perf_counter() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        broker.stop()
        front_cam.close()
        wrist_cam.close()
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    main()
