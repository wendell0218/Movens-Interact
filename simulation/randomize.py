import argparse
import copy
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from simulation.common import AppleSimulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-setup", default="assets/scene.json")
    parser.add_argument("--robot-config")
    parser.add_argument("--output", type=Path, default=Path("outputs/randomization"))
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = AppleSimulation(args.scene_setup, args.robot_config)
    source_hash = base.scene_info["scene_sha256"]
    tree = ET.parse(base.scene_path)
    apple_initial = np.fromstring(tree.find("./worldbody/body[@name='apple_root']").get("pos"), sep=" ")
    plate_initial = np.fromstring(tree.find("./worldbody/body[@name='plate_root']").get("pos"), sep=" ")
    cases = [("baseline", "baseline", (0, 0), (0, 0), "original"), ("apple_x_minus_01", "apple_position", (-.01, 0), (0, 0), "original"), ("apple_x_plus_01", "apple_position", (.01, 0), (0, 0), "original"), ("apple_y_minus_01", "apple_position", (0, -.01), (0, 0), "original"), ("apple_y_plus_01", "apple_position", (0, .01), (0, 0), "original"), ("plate_x_minus_02", "plate_position", (0, 0), (-.02, 0), "original"), ("plate_x_plus_02", "plate_position", (0, 0), (.02, 0), "original"), ("plate_y_minus_02", "plate_position", (0, 0), (0, -.02), "original"), ("plate_y_plus_02", "plate_position", (0, 0), (0, .02), "original"), ("table_dark_gray", "table_visual", (0, 0), (0, 0), "dark_gray"), ("table_blue", "table_visual", (0, 0), (0, 0), "blue"), ("combined", "combined", (.01, 0), (0, -.02), "dark_gray")]
    index = {"format": "movens_interact_randomization", "complete": False, "base_scene_sha256": source_hash, "source_scene_sha256": base.scene_info.get("source_scene_sha256"), "baseline_case_id": "baseline", "policy_seed": 42, "protocol": "Twelve predefined controlled cases. Only declared item XY or table visual material changes; no outcome-based tuning or physical material change.", "cases": []}
    images = []
    fields = ["body_mass", "body_inertia", "body_iquat", "body_quat", "body_gravcomp", "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity", "jnt_range", "dof_damping", "dof_armature", "dof_frictionloss", "actuator_gainprm", "actuator_biasprm", "actuator_gear", "cam_pos", "cam_quat", "cam_fovy"]
    for case_id, group, apple_delta, plate_delta, appearance in cases:
        scene_path = base.scene_path
        if case_id != "baseline":
            variant = copy.deepcopy(tree)
            variant.find("./worldbody/body[@name='apple_root']").set("pos", " ".join(str(float(x)) for x in apple_initial + np.r_[apple_delta, 0]))
            variant.find("./worldbody/body[@name='plate_root']").set("pos", " ".join(str(float(x)) for x in plate_initial + np.r_[plate_delta, 0]))
            compiler = variant.find("compiler")
            for node in variant.getroot().iter():
                if not node.get("file"):
                    continue
                path = Path(node.get("file"))
                if not path.is_absolute():
                    directory = compiler.get("meshdir", "") if node.tag == "mesh" else (compiler.get("texturedir", "") if node.tag == "texture" else "")
                    path = base.scene_path.parent / directory / path
                node.set("file", os.path.relpath(path.resolve(), output))
            if compiler is not None:
                compiler.attrib.pop("meshdir", None)
                compiler.attrib.pop("texturedir", None)
                compiler.attrib.pop("assetdir", None)
            if appearance != "original":
                material = variant.find("./asset/material[@name='table_PBR_Material']")
                if material is None:
                    raise ValueError("Visual variants require table_PBR_Material in the scene")
                material.attrib.pop("texture", None)
                material.set("rgba", "0.16 0.17 0.19 1" if appearance == "dark_gray" else "0.18 0.36 0.62 1")
                material.set("specular", "0.1")
                material.set("shininess", "0.1")
            scene_path = output / f"{case_id}.xml"
            ET.indent(variant, space="  ")
            variant.write(scene_path, encoding="unicode")
        metadata = copy.deepcopy(base.scene_info)
        metadata.update({"scene": os.path.relpath(scene_path, output), "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(), "robot_config": os.path.relpath(base.robot_path, output), "base_scene_sha256": source_hash, "case_id": case_id, "group": group, "changes": {"apple_delta_xyz": [*apple_delta, 0], "plate_delta_xyz": [*plate_delta, 0], "table_visual": appearance}, "seed": 42})
        setup_path = output / f"{case_id}.json"
        setup_path.write_text(json.dumps(metadata, indent=2) + "\n")
        sim = AppleSimulation(setup_path)
        checks = {field: bool(np.array_equal(getattr(base.model, field), getattr(sim.model, field))) for field in fields}
        positive_mass = base.model.body_mass > 0
        checks["mass_bearing_body_ipos"] = bool(np.array_equal(base.model.body_ipos[positive_mass], sim.model.body_ipos[positive_mass]))
        expected_position = base.model.body_pos.copy()
        expected_position[sim.apple_id, :2] += apple_delta
        expected_position[sim.plate_id, :2] += plate_delta
        checks["declared_item_translations_only"] = bool(np.allclose(expected_position, sim.model.body_pos, rtol=0, atol=1e-14))
        checks["physics_options"] = bool(np.array_equal(base.model.opt.gravity, sim.model.opt.gravity) and all(getattr(base.model.opt, key) == getattr(sim.model.opt, key) for key in ["timestep", "integrator", "solver", "iterations", "cone"]))
        if group == "table_visual":
            checks["visual_only_settled_qpos"] = bool(np.array_equal(base.data.qpos, sim.data.qpos))
        if not all(checks.values()):
            raise ValueError(f"Variant changed an undeclared physical parameter: {case_id}")
        metadata.update({"apple_xyz": sim.initial_world["apple_xyz"], "plate_xy": sim.initial_world["plate_xyz"][:2], "initial_positions_world": sim.initial_world, "post_settle_positions_world": sim.post_settle_world, "physical_parameter_checks": checks})
        setup_path.write_text(json.dumps(metadata, indent=2) + "\n")
        row = {"case_id": case_id, "group": group, "scene_json": setup_path.name, "scene_xml": os.path.relpath(scene_path, output), "scene_sha256": metadata["scene_sha256"], "changes": metadata["changes"], "seed": 42, "initial_positions_world": sim.initial_world, "post_settle_positions_world": sim.post_settle_world, "physical_parameter_checks": checks}
        if not args.no_images:
            image = Image.fromarray(sim.ego_rgb())
            image.save(output / f"{case_id}.initial_ego.png")
            draw = ImageDraw.Draw(image)
            draw.rectangle([0, 0, 480, 24], fill="white")
            draw.text((8, 7), case_id, fill="black")
            images.append(np.asarray(image))
            row["initial_ego_png"] = f"{case_id}.initial_ego.png"
        sim.close()
        index["cases"].append(row)
        print(json.dumps({"case_id": case_id, "compiled": True, "physical_parameters_verified": True}), flush=True)
    if images:
        Image.fromarray(np.concatenate([np.concatenate(images[k:k + 3], axis=1) for k in range(0, 12, 3)], axis=0)).save(output / "initial_ego_montage.png")
    index["complete"] = True
    index["source_scene_unchanged"] = hashlib.sha256(base.scene_path.read_bytes()).hexdigest() == source_hash
    if not index["source_scene_unchanged"]:
        raise ValueError("Source scene changed while constructing variants")
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    base.close()


if __name__ == "__main__":
    main()
