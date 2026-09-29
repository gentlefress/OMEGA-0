"""Inference client converting returned actions into Sonic's named inputs."""
from dataclasses import replace

import numpy as np

from .client import InferenceClient
from ..policies.geometry import load_skeleton
from ..transforms.geometry import forward_kinematics, rot6d_to_matrix, wrist_targets
from ..transforms.state import sonic_state


class OmegaInferenceClient(InferenceClient):
    """An inference client whose outputs match the Sonic command interface.

    Model layout is inferred from the server's action shape. Latent values are
    only sliced, never decoded.
    Normalized triggers remain inputs to Sonic's hand calibration.
    """

    def bind(self, ctx):
        super().bind(ctx)
        self.state_session = self.state_reference = None
        self.state_ports = tuple(port for port in ctx.inputs if port.startswith("state."))
        if self.state_ports:
            required = {"state.left_hand", "state.right_hand"}
            required |= {"state.lowstate"} if "state.lowstate" in ctx.inputs else {"state.joint_pos", "state.base_quat"}
            if set(self.state_ports) != required:
                raise ValueError(f"Sonic state inputs must be {sorted(required)}")
            # State is required even if a generic observation.required list
            # only named the image. Missing/stale state must not launch inference.
            self.selector.required.update(required)

    def preprocess(self, observation):
        observation = super().preprocess(observation)
        if not self.state_ports:
            return observation
        samples = dict(observation.samples)
        fields = {port.removeprefix("state."): samples.pop(port).value for port in self.state_ports}
        # This preprocessing runs on the inference worker, so its heading
        # reference follows that worker's session without racing runtime resets.
        if self.state_reference is None or self.state_session != self.worker_session:
            quat = np.asarray(fields['lowstate']).reshape(-1)[29:33] if 'lowstate' in fields else np.asarray(fields['base_quat']).reshape(-1)
            self.state_reference = quat.copy()
            self.state_session = self.worker_session
        state = sonic_state(fields, hand_scale=float(self.options.get("hand_state_scale", 1.0)),
                            reference_quat=self.state_reference)
        source = "state.lowstate" if "lowstate" in fields else "state.joint_pos"
        samples["state"] = replace(observation.samples[source], value=state)
        return replace(observation, samples=samples)

    def validate_spec(self):
        field = self.spec.fields.get(self.options.get("action_field", "action"))
        layouts = {(66,): "latent66", (71,): "smpl71", (75,): "smpl75", (86,): "joints86"}
        if field is None or field.dtype != "float32" or field.shape not in layouts:
            raise ValueError("Unsupported Sonic model action specification: expected float32 action of size 66, 71, 75, or 86")
        self.layout = layouts[field.shape]
        self.skeleton = None
        if self.options.get("skeleton"):
            self.skeleton = load_skeleton(self.options["skeleton"])
        if self.layout in {"smpl71", "smpl75"} and self.skeleton is None:
            raise ValueError("SMPL model output requires a calibrated skeleton")

    def command_fields(self):
        fields = {"token_state", "frame_index"} if self.layout == "latent66" else {
            "smpl_pose", "smpl_joints", "body_quat_w", "joint_pos", "joint_vel", "frame_index"}
        return fields | {"left_trigger", "right_trigger"}

    def _reset(self, ctx):
        self.history = []
        super()._reset(ctx)

    def prepare_command(self, command):
        values = command.values
        action = np.asarray(values[self.options.get("action_field", "action")], np.float32)
        expected = {"latent66": 66, "smpl71": 71, "smpl75": 75, "joints86": 86}[self.layout]
        if action.shape != (expected,):
            raise ValueError(f"{self.layout} requires shape {(expected,)}")
        if self.layout == "latent66":
            left, right = action[64:66]
            fields = {"token_state": action[:64][None], "frame_index": np.asarray([command.step], np.int64)}
        else:
            from scipy.spatial.transform import Rotation
            if self.layout == "smpl71":
                orientation = Rotation.from_matrix(rot6d_to_matrix(action[:6]))
                pose, left, right = action[6:69].reshape(21, 3), action[69], action[70]
            elif self.layout == "smpl75":
                orientation = Rotation.from_rotvec(action[:3])
                pose, left, right = action[3:66].reshape(21, 3), action[66:69].mean(), action[69:72].mean()
            else:
                orientation = Rotation.from_matrix(rot6d_to_matrix(action[80:]))
                pose, left, right = np.zeros((21, 3)), action[78], action[79]
            body = orientation * Rotation.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
            if self.layout == "joints86":
                joints = action[:72].reshape(24, 3)
                joint_pos = np.zeros(29, np.float32)
                joint_pos[-6:] = action[72:78]
            else:
                rest, parents = self.skeleton["rest_joints"], self.skeleton["parents"]
                angles = np.zeros((len(rest), 3))
                angles[0], angles[1:22] = orientation.as_rotvec(), pose
                joints_world = forward_kinematics(rest, parents, angles)[self.skeleton["output_indices"]]
                joints = body.inv().apply(joints_world)
                joint_pos = wrist_targets(pose)
            quat = body.as_quat()[[3, 0, 1, 2]]
            current = {"smpl_pose": pose, "smpl_joints": joints, "body_quat_w": quat, "joint_pos": joint_pos, "joint_vel": np.zeros(29)}
            size = int(self.options.get("history_size", 1))
            if size < 1:
                raise ValueError("History size must be positive")
            current["frame_index"] = np.asarray(command.step, np.int64)
            self.history.append(current)
            self.history = self.history[-size:]
            if self.options.get("history_start", "pad") == "wait" and len(self.history) < size:
                return None
            padded = [self.history[0]] * (size - len(self.history)) + self.history
            fields = {name: np.stack([item[name] for item in padded]).astype(np.float32) for name in current if name != "frame_index"}
            fields["frame_index"] = np.asarray([item["frame_index"] for item in padded], np.int64) if self.options.get("history_frame_indices", False) else np.asarray([command.step], np.int64)
        fields["left_trigger"] = np.asarray([left], np.float32)
        fields["right_trigger"] = np.asarray([right], np.float32)
        return replace(command, values=fields)
