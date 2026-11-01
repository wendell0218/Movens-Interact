import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

ISAAC_JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint"
)


def quaternion_matrix(quaternion):
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), dim=-1).reshape(quaternion.shape[:-1] + (3, 3))


class G1Kinematics(nn.Module):

    def __init__(self, xml_path, joint_names=None):
        super().__init__()
        xml_path = Path(xml_path).resolve()
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.joint_names = tuple(ISAAC_JOINT_NAMES if joint_names is None else joint_names)
        if len(self.joint_names) != 29 or len(set(self.joint_names)) != 29:
            raise ValueError("Exactly29 unique input body joint names are required")
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        if min(joint_ids) < 0:
            raise ValueError("An input joint name is missing from the compiled model")
        self.wrist_names = ("left_wrist_yaw_link", "right_wrist_yaw_link")
        wrists = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.wrist_names]
        if min(wrists) < 1:
            raise ValueError("The model must contain both wrist_yaw_link bodies")
        ancestors = set()
        for wrist in wrists:
            body_id = wrist
            while body_id:
                ancestors.add(body_id)
                body_id = int(model.body_parentid[body_id])
        bodies = sorted(ancestors)
        compact = {body: i for i, body in enumerate(bodies)}
        roots = [body for body in bodies if model.body_parentid[body] == 0]
        if len(roots) != 1:
            raise ValueError("Both wrists must belong to one root body tree")
        root = roots[0]
        self.parents = [compact.get(int(model.body_parentid[body]), -1) for body in bodies]
        self.wrist_indices = [compact[body] for body in wrists]
        self.body_joint_specs = []
        axes, anchors, references, joint_types, used = [], [], [], [], []
        root_free = None
        by_joint_id = {joint: i for i, joint in enumerate(joint_ids)}
        for body in bodies:
            specs = []
            for joint in range(int(model.body_jntadr[body]), int(model.body_jntadr[body] + model.body_jntnum[body])):
                kind = int(model.jnt_type[joint])
                if kind == mujoco.mjtJoint.mjJNT_FREE and body == root:
                    root_free = int(model.jnt_qposadr[joint])
                    continue
                if kind not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) or joint not in by_joint_id:
                    raise ValueError("Every non-root ancestor joint must be an input hinge/slide joint")
                specs.append((by_joint_id[joint], len(axes), kind))
                axes.append(model.jnt_axis[joint].copy())
                anchors.append(model.jnt_pos[joint].copy())
                references.append(float(model.qpos0[model.jnt_qposadr[joint]]))
                joint_types.append(kind)
                used.append(self.joint_names[by_joint_id[joint]])
            self.body_joint_specs.append(specs)
        rotations = []
        for body in bodies:
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, model.body_quat[body])
            rotations.append(matrix.reshape(3, 3))
        axes = np.asarray(axes, dtype=np.float64)
        skew = np.zeros((len(axes), 3, 3))
        skew[:, 0, 1], skew[:, 0, 2] = -axes[:, 2], axes[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = axes[:, 2], -axes[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -axes[:, 1], axes[:, 0]
        constants = {
            "body_positions": model.body_pos[bodies], "body_rotations": np.asarray(rotations),
            "joint_axes": axes, "joint_anchors": np.asarray(anchors), "joint_references": np.asarray(references),
            "joint_skew": skew, "joint_skew_squared": skew @ skew,
            "joint_limits": model.jnt_range[joint_ids],
            "default_root_xyz": model.qpos0[root_free:root_free + 3] if root_free is not None else model.body_pos[root],
            "default_root_quat_wxyz": model.qpos0[root_free + 3:root_free + 7] if root_free is not None else model.body_quat[root]
        }
        for name, value in constants.items():
            self.register_buffer(name, torch.as_tensor(np.asarray(value).copy(), dtype=torch.float64))
        self.register_buffer("joint_limited", torch.as_tensor(model.jnt_limited[joint_ids].astype(bool)))
        self.metadata = {
            "implementation": "g1_compiled_mujoco_tree_fk_v1", "xml_path": str(xml_path),
            "xml_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "mujoco_version": mujoco.__version__, "joint_names": list(self.joint_names),
            "wrist_names": list(self.wrist_names), "root_body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root),
            "root_pose_semantics": "World pose of compiled root body; WXYZ normalized quaternion",
            "position_units": "meter", "joint_units": "radian", "rotation_semantics": "body-to-world rotation matrices",
            "forward_precision": "Autocast disabled internally; float16/bfloat16/float32 inputs compute in float32, float64 inputs remain float64",
            "compiled_ancestor_body_count": len(bodies), "active_scalar_joint_count": len(axes),
            "input_joints_not_affecting_wrist_fk": [name for name in self.joint_names if name not in used],
            "joint_reference_values": dict(zip(used, references)), "joint_limits": model.jnt_range[joint_ids].tolist(),
            "official_reference": "https://mujoco.readthedocs.io/en/stable/overview.html#joint-reference"
        }

    def forward(self, q, root_xyz=None, root_quat_wxyz=None):

        if not q.is_floating_point():
            raise ValueError("q must be a floating tensor")
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=q.device.type, enabled=False):
            return self._forward(q.to(dtype=dtype), root_xyz, root_quat_wxyz)


    def _forward(self, q, root_xyz=None, root_quat_wxyz=None):
        if q.shape[-1] != 29 or not q.is_floating_point():
            raise ValueError("q must be a floating tensor ending in29 body joint coordinates")
        leading = q.shape[:-1]
        roots = []
        for value, default, width in [(root_xyz, self.default_root_xyz, 3),
                                       (root_quat_wxyz, self.default_root_quat_wxyz, 4)]:
            value = default.to(q) if value is None else torch.as_tensor(value, dtype=q.dtype, device=q.device)
            if value.shape[-1] != width:
                raise ValueError("Root pose coordinate dimension mismatch")
            if value.ndim == q.ndim - 1 and value.shape[:-1] == q.shape[:-2]:
                value = value.unsqueeze(-2)
            roots.append(torch.broadcast_to(value, leading + (width,)))
        root_position, root_quaternion = roots
        root_rotation = quaternion_matrix(root_quaternion)
        body_pos, body_rot = self.body_positions.to(q), self.body_rotations.to(q)
        axes, anchors, references = self.joint_axes.to(q), self.joint_anchors.to(q), self.joint_references.to(q)
        skew, skew2 = self.joint_skew.to(q), self.joint_skew_squared.to(q)
        eye = torch.eye(3, dtype=q.dtype, device=q.device)
        positions, rotations = [], []
        for body, parent in enumerate(self.parents):
            if parent < 0:
                position, rotation = root_position, root_rotation
            else:
                rotation = rotations[parent] @ body_rot[body]
                position = positions[parent] + (rotations[parent] @ body_pos[body, :, None]).squeeze(-1)
            for qindex, joint, kind in self.body_joint_specs[body]:
                angle = q[..., qindex] - references[joint]
                if kind == mujoco.mjtJoint.mjJNT_SLIDE:
                    position = position + (rotation @ axes[joint, :, None]).squeeze(-1) * angle[..., None]
                else:
                    turn = eye + angle.sin()[..., None, None] * skew[joint] + (1 - angle.cos())[..., None, None] * skew2[joint]
                    rotated = rotation @ turn
                    anchor = anchors[joint, :, None]
                    position = position + (rotation @ anchor - rotated @ anchor).squeeze(-1)
                    rotation = rotated
            positions.append(position)
            rotations.append(rotation)
        return (torch.stack([positions[i] for i in self.wrist_indices], dim=-2),
                torch.stack([rotations[i] for i in self.wrist_indices], dim=-3))
