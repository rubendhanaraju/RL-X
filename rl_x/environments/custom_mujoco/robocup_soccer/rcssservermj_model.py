import os
from pathlib import Path

import mujoco
import numpy as np
from dm_control import mjcf


SERVER_T1_NOMINAL_JOINT_POSITIONS = {
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "Left_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.4,
    "Right_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.4,
    "Waist": 0.0,
    "Left_Hip_Pitch": -0.4,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.8,
    "Left_Ankle_Pitch": -0.4,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.4,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.8,
    "Right_Ankle_Pitch": -0.4,
    "Right_Ankle_Roll": 0.0,
}


def rcssservermj_root(env_config):
    simulator_config = env_config.get("simulator", {})
    root = simulator_config.get("rcssservermj_root")
    if root:
        return Path(root)
    return Path(os.environ.get("RCSSSERVERMJ_ROOT", "/home/ruben/Documents/GitHub/rcssservermj"))


def uses_rcssservermj_model(env_config):
    return env_config.get("simulator", {}).get("model_source", "rlx") == "rcssservermj"


def build_rcssservermj_xml(env_config, *, object_type):
    root = rcssservermj_root(env_config)
    robot_path = root / "src" / "rcsssmj" / "resources" / "robots" / "T1" / "robot.xml"
    if not robot_path.is_file():
        raise FileNotFoundError(f"Could not find rcssservermj T1 model: {robot_path}")

    xml_handle = mjcf.from_path(robot_path.as_posix())
    add_server_training_world(xml_handle, object_type=object_type)
    if env_config.get("simulator", {}).get("disable_nonfoot_contacts", True):
        disable_nonfoot_robot_contacts(xml_handle)
    add_training_sensors(xml_handle)
    return xml_handle


def disable_nonfoot_robot_contacts(xml_handle):
    for geom in xml_handle.find_all("geom"):
        geom.contype = "0"
        geom.conaffinity = "0"

    add_contact_pair_if_missing(xml_handle, "pitch", "left_foot")
    add_contact_pair_if_missing(xml_handle, "pitch", "right_foot")
    if xml_handle.find("geom", "ball") is not None:
        add_contact_pair_if_missing(xml_handle, "pitch", "ball")
        add_contact_pair_if_missing(xml_handle, "left_foot", "ball")
        add_contact_pair_if_missing(xml_handle, "right_foot", "ball")


def add_contact_pair_if_missing(xml_handle, geom1, geom2):
    xml_handle.contact.add("pair", geom1=geom1, geom2=geom2)


def add_server_training_world(xml_handle, *, object_type):
    if xml_handle.find("geom", "pitch") is None:
        xml_handle.worldbody.add(
            "geom",
            name="pitch",
            pos="0 0 0",
            size="32 24 40",
            type="plane",
        )

    if object_type == "ball":
        if xml_handle.find("body", "ball") is None:
            ball = xml_handle.worldbody.add("body", name="ball", pos="1.0 0.0 0.11")
            ball.add("freejoint", name="ball-root")
            ball.add("site", name="B-vismarker", pos="0 0 0")
            ball.add(
                "geom",
                name="ball",
                pos="0 0 0",
                size="0.11",
                mass="0.41",
                friction="0.4 0.01 0.01",
                rgba="1 1 1 1",
                condim="6",
                priority="1",
                solref="-5000 -20",
                type="sphere",
            )
    elif object_type == "point":
        if xml_handle.find("body", "point") is None:
            point = xml_handle.worldbody.add("body", name="point", pos="1.0 0.0 0.05")
            point.add("freejoint", name="point-root")
            point.add(
                "geom",
                name="point",
                type="sphere",
                size="0.05",
                mass="0.01",
                contype="0",
                conaffinity="0",
                rgba="0.1 0.8 1.0 1",
            )
    else:
        raise ValueError(f"Unsupported server-fidelity object type: {object_type}")


def add_training_sensors(xml_handle):
    sensor = xml_handle.sensor

    if xml_handle.find("sensor", "torso_linear_velocity") is None:
        sensor.add("velocimeter", site="torso", name="torso_linear_velocity")
    if xml_handle.find("sensor", "left_foot_global_linear_velocity") is None:
        sensor.add(
            "framelinvel",
            objtype="site",
            objname="lfoot-vismarker",
            name="left_foot_global_linear_velocity",
        )
    if xml_handle.find("sensor", "right_foot_global_linear_velocity") is None:
        sensor.add(
            "framelinvel",
            objtype="site",
            objname="rfoot-vismarker",
            name="right_foot_global_linear_velocity",
        )


def home_qpos_from_model(model):
    data = mujoco.MjData(model)
    qpos = data.qpos.copy()
    for joint_name, joint_position in SERVER_T1_NOMINAL_JOINT_POSITIONS.items():
        qpos[model.joint(joint_name).qposadr[0]] = joint_position
    return qpos


def server_position_actuator_ids(model):
    return np.array(
        [
            actuator_id
            for actuator_id in range(model.nu)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id).endswith("_pos")
        ],
        dtype=int,
    )


def server_actuator_triplet_ids(model, position_actuator_ids):
    position_ids = []
    velocity_ids = []
    torque_ids = []
    for position_actuator_id in position_actuator_ids:
        pos_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, position_actuator_id)
        motor_name = pos_name[:-4]
        position_ids.append(position_actuator_id)
        velocity_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{motor_name}_vel"))
        torque_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{motor_name}_tau"))

    return (
        np.asarray(torque_ids, dtype=int),
        np.asarray(position_ids, dtype=int),
        np.asarray(velocity_ids, dtype=int),
    )


def set_server_pd_gains(model, position_actuator_ids, *, kp=25.0, kd=0.6):
    torque_ids, position_ids, velocity_ids = server_actuator_triplet_ids(model, position_actuator_ids)

    model.actuator_gainprm[position_ids, 0] = kp
    model.actuator_biasprm[position_ids, 1] = -kp
    model.actuator_gainprm[velocity_ids, 0] = kd
    model.actuator_biasprm[velocity_ids, 2] = -kd
    model.actuator_gainprm[torque_ids, 0] = 1.0


def server_joint_names_from_position_actuators(model, position_actuator_ids):
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[actuator_id, 0])
        for actuator_id in position_actuator_ids
    ]
