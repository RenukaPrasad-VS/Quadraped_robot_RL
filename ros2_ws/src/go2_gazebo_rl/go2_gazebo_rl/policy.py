"""Load the trained actor and the deploy config written by training.

Accepts either a raw rsl-rl checkpoint (``model_*.pt``; needs torch only to
read the file, inference is NumPy) or an exported ``policy.onnx``
(onnxruntime). Both reproduce the deterministic training-time action.
"""

from __future__ import annotations

import numpy as np
import yaml

# Unitree SDK joint order (also the order used in deploy.yaml's joint_ids_map).
SDK_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


class _TolerantLoader(yaml.SafeLoader):
    """deploy.yaml contains python-specific tags (tuple, slice); read them as plain data."""


def _python_tag(loader, suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/", _python_tag)


class DeployCfg:
    def __init__(self, path: str):
        with open(path) as f:
            cfg = yaml.load(f, Loader=_TolerantLoader)
        # joint_ids_map[i] = SDK index of the i-th joint in the policy's (Isaac Lab) order
        self.joint_names = [SDK_JOINTS[i] for i in cfg["joint_ids_map"]]
        self.step_dt = float(cfg["step_dt"])
        self.kp = np.array(cfg["stiffness"], dtype=np.float32)
        self.kd = np.array(cfg["damping"], dtype=np.float32)
        self.default_q = np.array(cfg["default_joint_pos"], dtype=np.float32)
        act = cfg["actions"]["JointPositionAction"]
        self.action_scale = np.array(act["scale"], dtype=np.float32)
        self.action_offset = np.array(act["offset"], dtype=np.float32)
        self.action_clip = np.array(act["clip"], dtype=np.float32) if act.get("clip") else None
        self.obs_terms = []  # (name, scale array, clip tuple|None) in policy input order
        for name, term in cfg["observations"].items():
            if term.get("history_length", 1) != 1:
                raise ValueError(f"observation history is not supported (term {name})")
            clip = tuple(term["clip"]) if term.get("clip") else None
            self.obs_terms.append((name, np.array(term["scale"], dtype=np.float32), clip))
        self.obs_dim = sum(len(s) for _, s, _ in self.obs_terms)


class Policy:
    def __init__(self, path: str):
        self.path = path
        if path.endswith(".onnx"):
            import onnxruntime as ort

            self._sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self._input = self._sess.get_inputs()[0].name
            self.obs_dim = self._sess.get_inputs()[0].shape[-1]
            self._run = self._run_onnx
        else:
            import torch

            sd = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
            g = lambda k: sd[k].numpy().astype(np.float32)  # noqa: E731
            self._mean = g("obs_normalizer._mean")[0] if "obs_normalizer._mean" in sd else None
            self._std = g("obs_normalizer._std")[0] if "obs_normalizer._std" in sd else None
            idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
            self._layers = [(g(f"mlp.{i}.weight"), g(f"mlp.{i}.bias")) for i in idx]
            self.obs_dim = self._layers[0][0].shape[1]
            self._run = self._run_numpy

    def _run_onnx(self, obs: np.ndarray) -> np.ndarray:
        return self._sess.run(None, {self._input: obs[None, :].astype(np.float32)})[0][0]

    def _run_numpy(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32)
        if self._mean is not None:
            x = (x - self._mean) / (self._std + 1e-2)  # rsl-rl EmpiricalNormalization
        for i, (w, b) in enumerate(self._layers):
            x = w @ x + b
            if i < len(self._layers) - 1:
                x = np.where(x > 0, x, np.expm1(x))  # ELU
        return x

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self._run(obs)
