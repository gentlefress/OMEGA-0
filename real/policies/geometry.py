"""Sonic/Pico coordinate conventions and three-point calibration.

Adapted from pico_manager_thread_server.py; PyTorch is loaded only for PKL assets.
"""
import json
from pathlib import Path
from xml.etree import ElementTree
import numpy as np


def load_skeleton(path):
    """Load the original Sonic tensor asset or an existing converted JSON file."""
    path = Path(path)
    if path.suffix.lower() == '.json':
        return json.loads(path.read_text())
    import torch
    asset = torch.load(path, map_location='cpu', weights_only=True)
    rest = np.asarray(asset['J']).reshape(-1, 3)
    parents = np.asarray(asset['parents_list'], dtype=np.int64).reshape(-1).copy()
    if rest.shape != (55, 3) or len(parents) != 55 or not np.isfinite(rest).all():
        raise ValueError('Expected Sonic SMPL-X rest joints and 55 parent indices')
    parents[0] = -1
    if any(not 0 <= parent < index for index, parent in enumerate(parents[1:], 1)):
        raise ValueError('Skeleton parents must precede children')
    return {'rest_joints': rest.tolist(), 'parents': parents.tolist(),
            'output_indices': [*range(22), 39, 54]}


def neck_targets(body):
    from scipy.spatial.transform import Rotation as R
    relative = R.from_quat(body[9, 3:]).inv() * R.from_quat(body[15, 3:])
    roll, pitch, _ = relative.as_euler("xyz")
    return 1.5 * np.array([-pitch, roll], np.float32) + np.array([0., -.7240389318822298], np.float32)


def quat_lerp(first, second, alpha):
    first, second = np.asarray(first), np.asarray(second)
    second = np.where((np.sum(first * second, axis=-1) < 0)[..., None], -second, second)
    value = (1. - alpha) * first + alpha * second
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def pose_lerp(first, second, alpha):
    from scipy.spatial.transform import Rotation as R
    return R.from_quat(quat_lerp(R.from_rotvec(first).as_quat(), R.from_rotvec(second).as_quat(), alpha)).as_rotvec()


def three_point_pose(body):
    from scipy.spatial.transform import Rotation as R
    body = np.asarray(body)
    if body.shape != (24, 7) or not np.isfinite(body).all():
        raise ValueError("Pico three-point input must be 24 xyz/xyzw poses")
    change = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0.]])
    positions = (body[:, :3] @ change.T).astype(np.float32)
    rotations = R.from_matrix(change @ R.from_quat(body[:, 3:]).as_matrix() @ change.T)
    # Preserve the source's intermediate float32 quaternion rounding.
    rotations = R.from_quat(rotations.as_quat().astype(np.float32))
    indices = [0, 22, 23, 12]
    offsets = R.from_euler("xyz", [[0, 0, -90], [90, 0, 0], [-90, 0, 180], [0, 0, -90]], degrees=True)
    key = R.from_quat((rotations[indices] * offsets).as_quat().astype(np.float32))
    root = key[0].inv()
    result = np.empty((3, 7), np.float32)
    result[:, :3] = root.apply(positions[indices[1:]] - positions[0])
    result[:, 3:] = (root * key[1:]).as_quat()[:, [3, 0, 1, 2]]
    return result


G1_JOINTS = [f"{side}_{joint}_joint" for side in ("left", "right")
             for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")] + [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] + [
    f"{side}_{joint}_joint" for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]


class G1Kinematics:
    """URDF FK for wrist calibration; mesh/physics packages are unnecessary."""
    def __init__(self, urdf, joint_names=None):
        from scipy.spatial.transform import Rotation as R
        tree = ElementTree.parse(urdf).getroot()
        self.joint_names = list(joint_names or G1_JOINTS)
        self.joints = []
        def vector(node, attribute, default):
            return np.fromstring(node.get(attribute, default) if node is not None else default, sep=" ")
        for joint in tree.findall("joint"):
            kind = joint.attrib["type"]
            if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
                raise ValueError(f"Unsupported calibration URDF joint: {kind}")
            origin = joint.find("origin")
            matrix = np.eye(4)
            matrix[:3, :3] = R.from_euler("xyz", vector(origin, "rpy", "0 0 0")).as_matrix()
            matrix[:3, 3] = vector(origin, "xyz", "0 0 0")
            self.joints.append((joint.attrib["name"], joint.find("parent").attrib["link"],
                                joint.find("child").attrib["link"], kind, matrix,
                                vector(joint.find("axis"), "xyz", "1 0 0")))
        names = {item[0] for item in self.joints}
        if not set(self.joint_names) <= names:
            raise ValueError("Calibration joint names are missing from the URDF")
        children = {item[2] for item in self.joints}
        self.roots = {item[1] for item in self.joints} - children

    def wrists(self, joints):
        from scipy.spatial.transform import Rotation as R
        values = np.asarray(joints).reshape(-1)
        if len(values) != len(self.joint_names):
            raise ValueError("Robot calibration joint dimensions differ")
        angles = dict(zip(self.joint_names, values))
        transforms = {name: np.eye(4) for name in self.roots}
        pending = list(self.joints)
        while pending:
            remaining = []
            for name, parent, child, kind, origin, axis in pending:
                if parent not in transforms:
                    remaining.append((name, parent, child, kind, origin, axis))
                    continue
                motion = np.eye(4)
                if kind in {"revolute", "continuous"}:
                    motion[:3, :3] = R.from_rotvec(axis * angles.get(name, 0.)).as_matrix()
                elif kind == "prismatic":
                    motion[:3, 3] = axis * angles.get(name, 0.)
                transforms[child] = transforms[parent] @ origin @ motion
            if len(remaining) == len(pending):
                raise ValueError("Disconnected or cyclic calibration URDF")
            pending = remaining
        result = np.empty((2, 7))
        for index, (side, y) in enumerate((("left", -.025), ("right", .025))):
            matrix = transforms[f"{side}_wrist_yaw_link"]
            result[index, :3] = matrix[:3, 3] + matrix[:3, :3] @ np.array([.18, y, 0.])
            result[index, 3:] = R.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
        return result


class ThreePointCalibration:
    def __init__(self):
        self.neck = self.positions = self.rotations = None

    def capture(self, body, robot_wrists, *, preserve_neck=False):
        from scipy.spatial.transform import Rotation as R
        raw = three_point_pose(body)
        if self.neck is None or not preserve_neck:
            self.neck = R.from_quat(raw[2, [4, 5, 6, 3]]).inv()
        corrected = self.neck * R.from_quat(raw[:2, 3:][:, [1, 2, 3, 0]])
        self.positions = self.neck.apply(raw[:2, :3]) - robot_wrists[:, :3]
        self.rotations = R.from_quat(robot_wrists[:, 3:][:, [1, 2, 3, 0]]) * corrected.inv()

    def apply(self, body):
        from scipy.spatial.transform import Rotation as R
        raw = three_point_pose(body)
        if self.neck is None:
            return raw
        result = raw.copy()
        result[:2, :3] = self.neck.apply(raw[:2, :3]) - self.positions
        wrists = self.rotations * self.neck * R.from_quat(raw[:2, 3:][:, [1, 2, 3, 0]])
        result[:2, 3:] = wrists.as_quat()[:, [3, 0, 1, 2]]
        neck = self.neck * R.from_quat(raw[2, [4, 5, 6, 3]])
        result[2, 3:] = neck.as_quat()[[3, 0, 1, 2]]
        result[2, :3] = np.array([0., 0., .05]) + .35 * neck.apply([0, 0, 1])
        return result
