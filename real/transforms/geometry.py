"""Explicit rotation conventions and calibrated forward kinematics, independent of Sonic."""
from __future__ import annotations

import numpy as np


def rot6d_to_matrix(value):
    """Row-major first two columns: [r00, r01, r10, r11, r20, r21]."""
    columns = np.asarray(value, dtype=np.float64).reshape(3, 2)
    first = columns[:, 0]
    norm = np.linalg.norm(first)
    if norm < 1e-12:
        raise ValueError("Degenerate first rotation column")
    first = first / norm
    second = columns[:, 1] - np.dot(first, columns[:, 1]) * first
    norm = np.linalg.norm(second)
    if norm < 1e-12:
        raise ValueError("Degenerate second rotation column")
    second /= norm
    return np.column_stack((first, second, np.cross(first, second)))


def swing_without_twist(rotvec, axis=(0, 1, 0)):
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_rotvec(rotvec)
    q = rotation.as_quat()  # xyzw
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    twist = np.r_[axis * np.dot(q[:3], axis), q[3]]
    if np.linalg.norm(twist) < 1e-12:
        return rotation
    return Rotation.from_quat(twist / np.linalg.norm(twist)).inv() * rotation


def wrist_targets(body_pose):
    """Preserve the existing G1 wrist mapping, with a defined zero-angle limit."""
    from scipy.spatial.transform import Rotation
    pose = np.asarray(body_pose).reshape(21, 3)
    left_wrist, right_wrist = Rotation.from_rotvec(pose[[19, 20]]).as_euler("XYZ")
    left = swing_without_twist(pose[17]).as_euler("XYZ") + left_wrist
    right = swing_without_twist(pose[18]).as_euler("XYZ") + right_wrist
    result = np.zeros(29, dtype=np.float32)
    # Pitch comes only from the wrist, not the elbow swing.
    result[[23, 25, 27]] = [left[0], left_wrist[1], left[2]]
    result[[24, 26, 28]] = [-right[0], -right_wrist[1], right[2]]
    return result


def forward_kinematics(rest_joints, parents, axis_angles):
    from scipy.spatial.transform import Rotation
    rest = np.asarray(rest_joints, dtype=np.float64)
    parents = list(parents)
    poses = np.asarray(axis_angles, dtype=np.float64)
    if rest.shape != poses.shape or rest.ndim != 2 or rest.shape[1] != 3 or len(parents) != len(rest):
        raise ValueError("Skeleton/pose dimensions differ")
    transforms = []
    rotations = Rotation.from_rotvec(poses).as_matrix()
    for index, parent in enumerate(parents):
        if index and (parent < 0 or parent >= index):
            raise ValueError("Skeleton parents must precede their children")
        transform = np.eye(4)
        transform[:3, :3] = rotations[index]
        transform[:3, 3] = rest[index] if index == 0 else rest[index] - rest[parent]
        transforms.append(transform if index == 0 else transforms[parent] @ transform)
    return np.stack(transforms)[:, :3, 3]
