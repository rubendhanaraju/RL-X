"""Static Booster T1 names and indices from ``booster_t1/data/plane.xml``."""

from __future__ import annotations

from enum import IntEnum


class T1Joint(IntEnum):
    """Named hinge-joint order after the free joint in the MJCF."""

    AA_HEAD_YAW = 0
    HEAD_PITCH = 1
    LEFT_SHOULDER_PITCH = 2
    LEFT_SHOULDER_ROLL = 3
    LEFT_ELBOW_PITCH = 4
    LEFT_ELBOW_YAW = 5
    RIGHT_SHOULDER_PITCH = 6
    RIGHT_SHOULDER_ROLL = 7
    RIGHT_ELBOW_PITCH = 8
    RIGHT_ELBOW_YAW = 9
    WAIST = 10
    LEFT_HIP_PITCH = 11
    LEFT_HIP_ROLL = 12
    LEFT_HIP_YAW = 13
    LEFT_KNEE_PITCH = 14
    LEFT_ANKLE_PITCH = 15
    LEFT_ANKLE_ROLL = 16
    RIGHT_HIP_PITCH = 17
    RIGHT_HIP_ROLL = 18
    RIGHT_HIP_YAW = 19
    RIGHT_KNEE_PITCH = 20
    RIGHT_ANKLE_PITCH = 21
    RIGHT_ANKLE_ROLL = 22


class T1Actuator(IntEnum):
    """Position-actuator order in the MJCF.

    The current MJCF actuator order matches the named hinge-joint order.
    """

    AA_HEAD_YAW = 0
    HEAD_PITCH = 1
    LEFT_SHOULDER_PITCH = 2
    LEFT_SHOULDER_ROLL = 3
    LEFT_ELBOW_PITCH = 4
    LEFT_ELBOW_YAW = 5
    RIGHT_SHOULDER_PITCH = 6
    RIGHT_SHOULDER_ROLL = 7
    RIGHT_ELBOW_PITCH = 8
    RIGHT_ELBOW_YAW = 9
    WAIST = 10
    LEFT_HIP_PITCH = 11
    LEFT_HIP_ROLL = 12
    LEFT_HIP_YAW = 13
    LEFT_KNEE_PITCH = 14
    LEFT_ANKLE_PITCH = 15
    LEFT_ANKLE_ROLL = 16
    RIGHT_HIP_PITCH = 17
    RIGHT_HIP_ROLL = 18
    RIGHT_HIP_YAW = 19
    RIGHT_KNEE_PITCH = 20
    RIGHT_ANKLE_PITCH = 21
    RIGHT_ANKLE_ROLL = 22


HEAD_ACTUATORS = (0, 1)
ARM_ACTUATORS = tuple(range(2, 10))
WAIST_ACTUATORS = (10,)
LEFT_LEG_ACTUATORS = tuple(range(11, 17))
RIGHT_LEG_ACTUATORS = tuple(range(17, 23))
LEG_ACTUATORS = LEFT_LEG_ACTUATORS + RIGHT_LEG_ACTUATORS

LEFT_LEG_JOINT_NAMES = (
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
)
RIGHT_LEG_JOINT_NAMES = (
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
)

LEFT_FOOT_SITE = "left_foot"
RIGHT_FOOT_SITE = "right_foot"
WAIST_BODY = "Waist"
