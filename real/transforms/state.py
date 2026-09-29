"""Assemble measured Sonic state for WAM requests; normalization is server-owned."""
import numpy as np


def sonic_state(fields, *, hand_scale=1.0, reference_quat=None):
    """29 joints, five channels per hand, then row-major rotation columns.

    Recorded lowstate is [29 joint positions, WXYZ quaternion, gyro, acceleration].
    Live telemetry supplies joint_pos/base_quat separately. Both produce the same
    raw 45-value state; recorded Inspire hands need hand_scale=0.001.
    """
    from scipy.spatial.transform import Rotation

    def vector(name, size=None):
        value = np.asarray(fields[name], np.float32).reshape(-1)
        if not np.isfinite(value).all() or (size is not None and value.size != size):
            raise ValueError(f"Invalid Sonic state {name}")
        return value

    if 'lowstate' in fields:
        lowstate = vector('lowstate', 39)
        joints, quat = lowstate[:29], lowstate[29:33]
    else:
        joints, quat = vector('joint_pos', 29), vector('base_quat', 4)
    left, right = vector('left_hand'), vector('right_hand')
    if min(left.size, right.size) < 5:
        raise ValueError("Sonic state requires at least five channels per hand")
    if not np.isfinite(hand_scale) or hand_scale <= 0:
        raise ValueError("Sonic hand state scale must be positive")
    if np.linalg.norm(quat) < 1e-8:
        raise ValueError("Sonic state has a zero base quaternion")
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
    if reference_quat is not None:
        reference = Rotation.from_quat(np.asarray(reference_quat)[[1, 2, 3, 0]]).as_matrix()
        yaw = np.arctan2(reference[1, 0], reference[0, 0])
        rotation = Rotation.from_rotvec([0., 0., -yaw]) * rotation
    rotation = rotation.as_matrix()[:, :2].reshape(-1)
    return np.concatenate([joints, left[:5] * hand_scale, right[:5] * hand_scale,
                           rotation]).astype(np.float32)
