import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from dm_control import mjcf


DEFAULT_RCSSSERVERMJ_ROOT = "/home/ruben/Documents/GitHub/RoboCup/rcssservermj"
RCSSSERVERMJ_ROBOT_XML = Path("src") / "rcsssmj" / "resources" / "robots" / "T1" / "robot.xml"
RCSSSERVERMJ_PACKAGE_DIR = Path("src") / "rcsssmj"


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
    candidates = []

    configured_root = simulator_config.get("rcssservermj_root")
    if configured_root:
        candidates.append(Path(configured_root).expanduser())

    env_root = os.environ.get("RCSSSERVERMJ_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    repo_root = Path(__file__).resolve().parents[4]
    candidates.extend(
        [
            Path.cwd() / "rcssservermj",
            Path.cwd().parent / "rcssservermj",
            repo_root / "rcssservermj",
            repo_root.parent / "rcssservermj",
            repo_root.parent / "RoboCup" / "rcssservermj",
            Path(DEFAULT_RCSSSERVERMJ_ROOT).expanduser(),
        ]
    )

    tried = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in tried:
            continue
        tried.append(candidate)
        if (candidate / RCSSSERVERMJ_ROBOT_XML).is_file() or (candidate / RCSSSERVERMJ_PACKAGE_DIR).is_dir():
            return candidate

    tried_paths = "\n".join(f"  - {candidate}" for candidate in tried)
    raise FileNotFoundError(
        "Could not find rcssservermj. Set RCSSSERVERMJ_ROOT or "
        "--environment.simulator.rcssservermj_root to the rcssservermj checkout.\n"
        f"Tried:\n{tried_paths}"
    )


def uses_rcssservermj_model(env_config):
    return env_config.get("simulator", {}).get("model_source", "rlx") == "rcssservermj"


def build_rcssservermj_xml(env_config, *, object_type):
    root = rcssservermj_root(env_config)
    world_source = env_config.get("simulator", {}).get("world_source", "server")
    if world_source == "server":
        xml_handle = build_server_soccer_world_xml(root, env_config, object_type=object_type)
        disable_mjx_unsupported_world_contacts(xml_handle)
        if env_config.get("simulator", {}).get("disable_nonfoot_contacts", False):
            disable_nonfoot_robot_contacts(xml_handle)
    elif world_source == "rlx":
        xml_handle = build_rlx_training_world_xml(root, object_type=object_type)
    else:
        raise ValueError(f"Unsupported rcssservermj world_source: {world_source}")

    add_training_sensors(xml_handle)
    return xml_handle


def disable_mjx_unsupported_world_contacts(xml_handle):
    # MJX cannot compile some of the far-away server goal/outer-floor contact
    # pairs (for example cylinder-box). These geoms are not part of the local
    # dribbling/point-tracking training dynamics, so keep them visual only.
    for geom in xml_handle.find_all("geom"):
        name = geom.name or ""
        if name.startswith("goal-") or name.endswith("-floor"):
            geom.contype = "0"
            geom.conaffinity = "0"


def build_server_soccer_world_xml(root, env_config, *, object_type):
    spec = build_server_soccer_world_spec(root, env_config, object_type=object_type)
    xml = sanitize_mjspec_xml(spec.to_xml())
    assets = server_assets(root)
    return mjcf.from_xml_string(xml, assets=assets)


def build_server_soccer_world_spec(root, env_config, *, object_type):
    root = Path(root).expanduser().resolve()
    ensure_rcssservermj_import(root)

    from rcsssmj.games.soccer.game_phase import GamePhase
    from rcsssmj.games.soccer.sim.soccer_referee import NoOpSoccerReferee
    from rcsssmj.games.soccer.sim.soccer_sim import SoccerSimulation
    from rcsssmj.games.soccer.soccer_fields import create_soccer_field
    from rcsssmj.games.soccer.soccer_rules import SoccerRuleBooks, create_soccer_rule_book

    simulator_config = env_config.get("simulator", {})
    field_name = simulator_config.get("field", "fifa7vs7")
    rules_name = simulator_config.get("rules", SoccerRuleBooks.SSIM.value)

    simulation = SoccerSimulation(
        field=create_soccer_field(field_name),
        rules=create_soccer_rule_book(rules_name),
        referee=NoOpSoccerReferee(),
        initial_game_phase=GamePhase.FIRST_HALF,
        enable_cheats=True,
    )
    spec = simulation._create_world()

    # Keep the soccer world/options from the actual server, but put the robot
    # free joint first because the RL-X env code stores the robot root at qpos[:7].
    spec.delete(spec.body("ball"))

    robot_spec = simulation.spec_provider.load_robot_spec("T1")
    if robot_spec is None:
        raise FileNotFoundError(f"Could not load rcssservermj T1 model from {root}")

    frame = spec.worldbody.add_frame()
    frame.attach_body(robot_spec.body("torso"), "", "")

    if object_type == "none":
        pass
    elif object_type == "ball":
        add_server_ball_to_spec(spec)
    elif object_type == "point":
        add_training_point_to_spec(spec)
    else:
        raise ValueError(f"Unsupported server-fidelity object type: {object_type}")

    return spec


def build_rlx_training_world_xml(root, *, object_type):
    robot_path = root / RCSSSERVERMJ_ROBOT_XML
    if not robot_path.is_file():
        raise FileNotFoundError(f"Could not find rcssservermj T1 model: {robot_path}")

    xml_handle = mjcf.from_path(robot_path.as_posix())
    xml_handle.option.density = 1.2
    xml_handle.option.viscosity = 2e-5
    add_rlx_training_world(xml_handle, object_type=object_type)
    disable_nonfoot_robot_contacts(xml_handle)
    return xml_handle


def ensure_rcssservermj_import(root):
    src_path = (Path(root) / "src").resolve()
    package_path = src_path / "rcsssmj"
    if not package_path.is_dir():
        raise FileNotFoundError(f"Could not find rcssservermj package: {package_path}")

    src_path_str = src_path.as_posix()
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)

    import rcsssmj

    loaded_path = Path(rcsssmj.__file__).resolve()
    try:
        loaded_path.relative_to(src_path)
    except ValueError as exc:
        raise ImportError(
            "rcsssmj was already imported from a different checkout: "
            f"{loaded_path}. Restart the process or set simulator.rcssservermj_root "
            f"to the loaded checkout."
        ) from exc


def sanitize_mjspec_xml(xml):
    root = ET.fromstring(xml)
    flatten_empty_default_classes(root)
    return ET.tostring(root, encoding="unicode")


def flatten_empty_default_classes(parent):
    for child in list(parent):
        flatten_empty_default_classes(child)

    if parent.tag != "default":
        return

    for child in list(parent):
        if child.tag == "default" and not child.attrib:
            insert_at = list(parent).index(child)
            parent.remove(child)
            for grandchild in list(child):
                child.remove(grandchild)
                parent.insert(insert_at, grandchild)
                insert_at += 1


def server_assets(root):
    root = Path(root)
    asset_dirs = [
        root / "src" / "rcsssmj" / "resources" / "environments" / "soccer" / "assets",
        root / "src" / "rcsssmj" / "resources" / "robots" / "T1" / "meshes",
    ]
    assets = {}
    for asset_dir in asset_dirs:
        if not asset_dir.is_dir():
            raise FileNotFoundError(f"Could not find rcssservermj asset directory: {asset_dir}")
        for asset_path in asset_dir.iterdir():
            if asset_path.is_file():
                assets[asset_path.name] = asset_path.read_bytes()
    return assets


def add_server_ball_to_spec(spec):
    ball = spec.worldbody.add_body()
    ball.name = "ball"
    ball.pos = (0.0, 0.0, 0.11)
    ball.add_freejoint(name="ball-root")
    site = ball.add_site()
    site.name = "B-vismarker"
    site.pos = (0.0, 0.0, 0.0)
    geom = ball.add_geom()
    geom.name = "ball"
    geom.pos = (0.0, 0.0, 0.0)
    geom.size = (0.11, 0.0, 0.0)
    geom.mass = 0.41
    geom.friction = (0.4, 0.01, 0.01)
    geom.rgba = (1.0, 1.0, 1.0, 1.0)
    geom.condim = 6
    geom.priority = 1
    geom.solref = (-5000.0, -20.0)
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.material = "ball"


def add_training_point_to_spec(spec):
    point = spec.worldbody.add_body()
    point.name = "point"
    point.pos = (1.0, 0.0, 0.05)
    point.add_freejoint(name="point-root")
    geom = point.add_geom()
    geom.name = "point"
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size = (0.05, 0.0, 0.0)
    geom.mass = 0.01
    geom.contype = 0
    geom.conaffinity = 0
    geom.rgba = (0.1, 0.8, 1.0, 1.0)


def add_rlx_training_world(xml_handle, *, object_type):
    if xml_handle.find("geom", "pitch") is None:
        xml_handle.worldbody.add(
            "geom",
            name="pitch",
            pos="0 0 0",
            size="32 24 40",
            type="plane",
            friction="1 0.01 0.005",
            solimp="0.015 1 0.015 0.5 2",
        )

    if object_type == "none":
        pass
    elif object_type == "ball":
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
        raise ValueError(f"Unsupported lightweight training object type: {object_type}")


def disable_nonfoot_robot_contacts(xml_handle):
    for geom in xml_handle.find_all("geom"):
        geom.contype = "0"
        geom.conaffinity = "0"

    add_contact_pair_if_missing(xml_handle, "pitch", "left_foot", kind="floor_foot")
    add_contact_pair_if_missing(xml_handle, "pitch", "right_foot", kind="floor_foot")
    add_contact_pair_if_missing(xml_handle, "left_hand", "left_thigh", kind="self")
    add_contact_pair_if_missing(xml_handle, "right_hand", "right_thigh", kind="self")
    if xml_handle.find("geom", "ball") is not None:
        add_contact_pair_if_missing(xml_handle, "pitch", "ball", kind="ball")
        add_contact_pair_if_missing(xml_handle, "left_foot", "ball", kind="ball")
        add_contact_pair_if_missing(xml_handle, "right_foot", "ball", kind="ball")


def add_contact_pair_if_missing(xml_handle, geom1, geom2, *, kind):
    # Explicit pairs do not inherit the same resolved params as automatic geom
    # contacts. Set the server-resolved values so contact reduction changes only
    # which contacts exist, not the physics of the contacts we keep.
    if kind == "floor_foot":
        xml_handle.contact.add(
            "pair",
            geom1=geom1,
            geom2=geom2,
            condim=3,
            friction=(1.0, 1.0, 0.01, 0.005, 0.005),
            solref=(0.02, 1.0),
            solimp=(0.015, 1.0, 0.015, 0.5, 2.0),
        )
    elif kind == "ball":
        xml_handle.contact.add(
            "pair",
            geom1=geom1,
            geom2=geom2,
            condim=6,
            friction=(0.4, 0.4, 0.01, 0.01, 0.01),
            solref=(-5000.0, -20.0),
            solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        )
    elif kind == "self":
        xml_handle.contact.add(
            "pair",
            geom1=geom1,
            geom2=geom2,
            condim=3,
            friction=(1.0, 1.0, 0.01, 0.005, 0.005),
            solref=(0.02, 1.0),
            solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        )
    else:
        raise ValueError(f"Unknown reduced contact pair kind: {kind}")


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
