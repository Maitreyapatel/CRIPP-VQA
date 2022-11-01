from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils

# from tdw.collisions import Collisions
from tdw.output_data import OutputData, Bounds, Images
from tdw.librarian import ModelLibrarian
import tdw.output_data as output_data

# from tdw.collisions import Collisions

import random
import math
import pandas as pd
import numpy as np
import json

import argparse

parser = argparse.ArgumentParser(description="Few required data generation parameters")
parser.add_argument(
    "-jold",
    "--jsonold",
    type=str,
    required=True,
    help="sample json file path with name",
)

args = parser.parse_args()


with open(args.jsonold, "r") as handle:
    data = json.load(handle)


# Create the avatar.
c = Controller()

resp = c.communicate(
    [
        {"$type": "set_render_quality", "render_quality": 5},
        {"$type": "set_screen_size", "width": 512, "height": 512},
        TDWUtils.create_empty_room(25, 25),
        {"$type": "create_avatar", "type": "A_Img_Caps_Kinematic", "id": "a"},
        {"$type": "set_target_framerate", "framerate": 25},
        {"$type": "set_pass_masks", "avatar_id": "a", "pass_masks": ["_img"]},
        {
            "$type": "teleport_avatar_to",
            "avatar_id": "a",
            "position": {"x": 0, "y": 10, "z": 10},
        },
        {
            "$type": "look_at_position",
            "avatar_id": "a",
            "position": {"x": 0, "y": 0, "z": 0},  
        },
    ]
)

# Disable post-processing, so the changes to the material aren't blurry.
# c.communicate({"$type": "set_post_process",
# 					"value": False})

c.communicate(
    [
        {"$type": "adjust_directional_light_intensity_by", "intensity": 0.25},
        {"$type": "set_anti_aliasing", "mode": "none"},
        {"$type": "rotate_directional_light_by", "angle": 20, "axis": "yaw"},
        {"$type": "rotate_directional_light_by", "angle": 65, "axis": "pitch"},
        {"$type": "set_shadow_strength", "strength": 10},
    ]
)

all_idx = []
static_objs = []
for i, obj in enumerate(data["initial_static_objects"]):
    all_idx.append(i)
    tmp = {
        "id": i,
        "name": obj["name"],
        "color": obj["color"],
        "material": obj["material"],
        "scale": obj["scale"],
        "loc": obj["loc"],
        "mass": obj["mass"],
    }

    static_objs.append(tmp)

dynamic_objs = []
for i, obj in enumerate(data["iniital_dynamic_objects"]):
    all_idx.append(len(data["initial_static_objects"]) + i)
    tmp = {
        "id": i + len(data["initial_static_objects"]),
        "name": obj["name"],
        "color": obj["color"],
        "material": obj["material"],
        "scale": obj["scale"],
        "loc": obj["loc"],
        "mass": obj["mass"],
        "target_object_id": obj["target_object_id"],
        "directional_force": obj["directional_force"],
        "initial_acc": obj["initial_acc"],
    }

    dynamic_objs.append(tmp)


# initialize static objects
for obj1 in static_objs:
    record = ModelLibrarian("models_flex.json").get_record(obj1["name"])
    c.communicate(
        {
            "$type": "add_object",
            "name": obj1["name"],
            "url": record.get_url(),
            "scale_factor": obj1["scale"],
            "position": obj1["loc"],
            "rotation": TDWUtils.VECTOR3_ZERO,
            "category": record.wcategory,
            "id": obj1["id"],
        }
    )

    c.communicate(
        TDWUtils.set_visual_material(
            c, record.substructure, obj1["id"], obj1["material"], quality="low"
        )
    )

    c.communicate(
        [
            {"$type": "set_mass", "id": obj1["id"], "mass": obj1["mass"]},
            {
                "$type": "set_physic_material",
                "id": obj1["id"],
                "dynamic_friction": 0,
                "static_friction": 0,
            },
            {"$type": "set_color", "color": obj1["color"], "id": obj1["id"]},
        ]
    )


# initialize dynamic objects
for obj2 in dynamic_objs:
    record = ModelLibrarian("models_flex.json").get_record(obj2["name"])
    c.communicate(
        {
            "$type": "add_object",
            "name": obj2["name"],
            "url": record.get_url(),
            "scale_factor": obj2["scale"],
            "position": obj2["loc"],
            "rotation": TDWUtils.VECTOR3_ZERO,
            "category": record.wcategory,
            "id": obj2["id"],
        }
    )

    c.communicate(
        TDWUtils.set_visual_material(
            c, record.substructure, obj2["id"], obj2["material"], quality="low"
        )
    )
    c.communicate(
        [
            {"$type": "set_mass", "id": obj2["id"], "mass": obj2["mass"]},
            {
                "$type": "set_physic_material",
                "id": obj2["id"],
                "dynamic_friction": 0,
                "static_friction": 0,
            },
            {"$type": "set_color", "color": obj2["color"], "id": obj2["id"]},
            {
                "$type": "apply_force_to_object",
                "force": obj2["directional_force"],
                "id": obj2["id"],
            },
        ]
    )


## Get the output data
c.communicate(
    [
        {"$type": "send_images", "frequency": "always"},
        {
            "$type": "send_bounds",
            "ids": [i for i in range(data["total_objects"])],
            "frequency": "always",
        },
        {
            "$type": "send_rigidbodies",
            "ids": [i for i in range(data["total_objects"])],
            "frequency": "always",
        },
        {
            "$type": "send_transforms",
            "ids": [i for i in range(data["total_objects"])],
            "frequency": "always",
        },
        {
            "$type": "send_collisions",
            "collision_types": ["obj", "env"],
            "enter": True,
            "exit": True,
            "stay": False,
        },
    ]
)


output_json = {
    "total_objects": data["total_objects"],
    "total_initial_moving_objects": data["total_initial_moving_objects"],
    "initial_static_objects": static_objs,
    "iniital_dynamic_objects": dynamic_objs,
    "frames": [],
    "collisions": [],
}


for i in range(125):  # number of frames
    resp = c.communicate([])
    images = Images(resp[0])
    TDWUtils.save_images(images, filename="{:04d}".format(i), output_directory="./dist")

    positions = output_data.Bounds(resp[1])
    vel = output_data.Rigidbodies(resp[2])
    rot = output_data.Transforms(resp[3])
    r_id = OutputData.get_data_type_id(resp[4])

    tmp_frame = {
        "frame": i,
    }
    for j in range(data["total_objects"]):
        tmp_frame[j] = {
            "location": positions.get_bottom(j),
            "velocity": vel.get_velocity(j),
            "rotation": rot.get_rotation(j),
        }
    output_json["frames"].append(tmp_frame)


c.communicate({"$type": "terminate"})
