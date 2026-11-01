import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class AppleSimulation:
    def __init__(self, scene_setup="assets/scene.json", robot_config=None, reset_protocol=None):
        setup_path = Path(scene_setup)
        if not setup_path.is_absolute():
            setup_path = ROOT / setup_path
        self.setup_path = setup_path.resolve()
        self.scene_info = json.loads(self.setup_path.read_text())
        scene_path = Path(self.scene_info["scene"])
        if not scene_path.is_absolute():
            scene_path = self.setup_path.parent / scene_path
        self.scene_path = scene_path.resolve()
        scene_hash = hashlib.sha256(self.scene_path.read_bytes()).hexdigest()
        if self.scene_info.get("scene_sha256") and scene_hash != self.scene_info["scene_sha256"]:
            raise ValueError("Scene file does not match its metadata checksum")
        self.scene_info["scene"] = str(self.scene_path)
        self.scene_info["scene_sha256"] = scene_hash
        if robot_config is None:
            robot_path = self.setup_path.parent / self.scene_info.get("robot_config", "robot.json")
        else:
            robot_path = Path(robot_config)
            if not robot_path.is_absolute():
                robot_path = ROOT / robot_path
        self.robot_path = robot_path.resolve()
        robot = json.loads(self.robot_path.read_text())
        self.body_names = robot["body_joint_names"]
        self.hand_names = robot["hand_joint_names"]
        self.joint_names = self.body_names + self.hand_names
        if len(self.body_names) != 29 or len(self.hand_names) != 14:
            raise ValueError("Expected 29 body joints and 14 hand joints")
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.vis.quality.offsamples = 0
        self.joint_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names])
        self.actuator_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.joint_names])
        if np.any(self.joint_ids < 0) or np.any(self.actuator_ids < 0) or len(set(self.joint_ids)) != 43:
            raise ValueError("Scene joint or actuator names do not match the robot configuration")
        self.qpos_indices = self.model.jnt_qposadr[self.joint_ids]
        self.joint_limits = self.model.jnt_range[self.joint_ids]
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.apple_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.scene_info["apple_body"])
        self.plate_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.scene_info["plate_body"])
        apple_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.scene_info["apple_joint"])
        if min(self.pelvis_id, self.apple_id, self.plate_id, apple_joint) < 0 or self.model.jnt_type[apple_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Scene must provide the named robot, free apple and plate")
        self.apple_dof = int(self.model.jnt_dofadr[apple_joint])
        self.robot_body_ids = {self.pelvis_id}
        self.apple_body_ids = {self.apple_id}
        self.plate_body_ids = {self.plate_id}
        table_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.scene_info.get("table_body", "table_root"))
        self.table_body_ids = {table_id} if table_id >= 0 else set()
        for body_id in range(self.model.nbody):
            for body_set in [self.robot_body_ids, self.apple_body_ids, self.plate_body_ids, self.table_body_ids]:
                if int(self.model.body_parentid[body_id]) in body_set:
                    body_set.add(body_id)
        self.apple_weight_N = float(self.model.body_subtreemass[self.apple_id] * -self.model.opt.gravity[2])
        self.substeps = int(round(.02 / self.model.opt.timestep))
        if abs(self.substeps * self.model.opt.timestep - .02) > 1e-10:
            raise ValueError("Physics timestep must divide the 20ms control interval")
        self.renderer = None
        self.default_reset = robot["reset_protocol"]
        self.reset(reset_protocol)

    def synchronize(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def reset(self, reset_protocol=None):
        protocol = copy.deepcopy(self.default_reset if reset_protocol is None else reset_protocol)
        order = [protocol["joint_names"].index(name) for name in self.joint_names]
        self.reset_protocol = protocol
        self.data = mujoco.MjData(self.model)
        self.data.qpos[self.qpos_indices] = np.asarray(protocol["qpos43"], dtype=np.float64)[order]
        self.data.ctrl[self.actuator_ids] = np.asarray(protocol["ctrl43"], dtype=np.float64)[order]
        mujoco.mj_forward(self.model, self.data)
        self.initial_world = {"apple_xyz": self.data.xpos[self.apple_id].tolist(), "apple_quaternion_wxyz": self.data.xquat[self.apple_id].tolist(), "plate_xyz": self.data.xpos[self.plate_id].tolist(), "plate_quaternion_wxyz": self.data.xquat[self.plate_id].tolist()}
        for _ in range(protocol["settle_steps"]):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        self.synchronize()
        self.baseline_apple_z = float(self.data.xpos[self.apple_id, 2])
        self.post_settle_world = {"apple_xyz": self.data.xpos[self.apple_id].tolist(), "apple_quaternion_wxyz": self.data.xquat[self.apple_id].tolist(), "plate_xyz": self.data.xpos[self.plate_id].tolist(), "plate_quaternion_wxyz": self.data.xquat[self.plate_id].tolist()}
        return self.observe()

    def observe(self):
        self.synchronize()
        body = np.r_[self.data.qpos[self.qpos_indices[:29]], self.data.xpos[self.pelvis_id], self.data.xquat[self.pelvis_id]].astype(np.float32)
        hand = self.data.qpos[self.qpos_indices[29:]].astype(np.float32)
        return body[None], hand[None]

    def render(self, camera=None):
        self.synchronize()
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=360, width=480)
        if camera is None:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [.35, 0, .95 + self.scene_info["root_z"] - .793]
            camera.distance, camera.azimuth, camera.elevation = 2.15, 135, -22
        self.renderer.update_scene(self.data, camera=camera)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        return self.renderer.render().copy()

    def ego_rgb(self):
        return self.render(self.scene_info["camera"]["name"])

    def step(self, target43):
        requested = np.asarray(target43, dtype=np.float64)
        if requested.shape != (43,) or not np.isfinite(requested).all():
            raise ValueError("Control target must contain 43 finite joint angles")
        target = np.clip(requested, self.joint_limits[:, 0], self.joint_limits[:, 1])
        self.data.ctrl[self.actuator_ids] = target
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        contacts = {key: False for key in ["plate", "table", "robot", "other", "left_fingers", "left_wrist"]}
        forces = {key: np.zeros(3) for key in ["plate", "table", "robot", "other"]}
        for index, contact in enumerate(self.data.contact[:self.data.ncon]):
            bodies = [int(self.model.geom_bodyid[g]) for g in [contact.geom1, contact.geom2]]
            names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or "" for body in bodies]
            apple_first = bodies[0] in self.apple_body_ids
            if not apple_first and bodies[1] not in self.apple_body_ids:
                continue
            if apple_first and bodies[1] in self.apple_body_ids:
                continue
            other = 1 if apple_first else 0
            name = names[other]
            kind = "plate" if bodies[other] in self.plate_body_ids else ("table" if bodies[other] in self.table_body_ids else ("robot" if bodies[other] in self.robot_body_ids else "other"))
            contacts[kind] = True
            contacts["left_fingers"] |= name.startswith("left_hand_")
            contacts["left_wrist"] |= name.startswith("left_wrist_")
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            world_force = contact.frame.reshape(3, 3).T @ force[:3]
            forces[kind] += -world_force if apple_first else world_force
        constraint = self.data.qfrc_constraint[self.apple_dof:self.apple_dof + 3].copy()
        body, hand = self.observe()
        finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        failure = None if finite else "non_finite_simulation_state"
        if finite and np.abs(self.data.qvel).max() > 200:
            failure = "simulation_velocity_diverged"
        return {"time": float(self.data.time), "force_evaluation_time": float(self.data.time - self.model.opt.timestep),
            "body_abs": body[0], "hand_state": hand[0], "apple_xyz": self.data.xpos[self.apple_id].copy(),
            "apple_quaternion_wxyz": self.data.xquat[self.apple_id].copy(), "plate_xyz": self.data.xpos[self.plate_id].copy(),
            "plate_xy_distance": float(np.linalg.norm(self.data.xpos[self.apple_id, :2] - self.data.xpos[self.plate_id, :2])),
            "contacts": contacts, "forces": forces, "constraint_force": constraint,
            "requested_target": requested.copy(), "target": target.copy(), "clipped": np.abs(target - requested) > 1e-7,
            "qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(), "ctrl": self.data.ctrl.copy(), "failure": failure}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def support_summary(samples, baseline_apple_z, apple_weight_N):
    if not samples:
        return {"support_verified_success": False, "grasped_and_lifted": False, "stable_place": False, "sample_count": 0}
    current = longest = 0.0
    for sample in samples:
        lifted = sample["contacts"]["left_fingers"] and not sample["contacts"]["table"] and sample["apple_xyz"][2] >= baseline_apple_z + .02
        current = current + .02 if lifted else 0.0
        longest = max(longest, current)
    recent = samples[-50:]
    positions = np.asarray([sample["apple_xyz"] for sample in recent])
    speed = float(np.linalg.norm(np.diff(positions, axis=0) * 50, axis=1).max()) if len(positions) > 1 else 0.0
    fractions = {key: float(np.mean([sample["contacts"][key] for sample in recent])) for key in ["plate", "table", "robot", "other", "left_fingers", "left_wrist"]}
    plate_force = np.asarray([sample["forces"]["plate"] for sample in recent])
    ratio = float(plate_force[:, 2].mean() / apple_weight_N)
    distance = max(sample["plate_xy_distance"] for sample in recent)
    z_span = float(np.ptp(positions[:, 2]))
    stable = bool(len(recent) >= 50 and fractions["plate"] == 1 and fractions["table"] == 0 and fractions["robot"] == 0 and fractions["other"] == 0 and speed < .02 and z_span < .005 and distance < .0541 and .95 < ratio < 1.05)
    peak_lift = max(float(sample["apple_xyz"][2]) for sample in samples) - baseline_apple_z
    grasped = bool(longest >= .2 and peak_lift >= .02)
    finite = all(sample["failure"] is None for sample in samples)
    return {"support_verified_success": bool(grasped and stable and finite), "grasped_and_lifted": grasped,
        "stable_place": stable, "sample_count": len(samples), "peak_lift_m": float(peak_lift),
        "longest_hand_contact_lift_ge2cm_off_table_seconds": longest,
        "final_apple_xyz": samples[-1]["apple_xyz"].tolist(), "plate_xy_distance_final_m": samples[-1]["plate_xy_distance"],
        "final_support_force_verification": {"duration_seconds": len(recent) / 50, "contact_fractions": fractions,
            "apple_weight_N": apple_weight_N, "mean_plate_force_world_xyz_N": plate_force.mean(axis=0).tolist(),
            "mean_plate_vertical_force_over_weight": ratio, "apple_max_speed_m_s": speed,
            "apple_z_span_m": z_span, "apple_plate_max_center_distance_m": distance,
            "approximate_center_containment_radius_m": .0541}}
