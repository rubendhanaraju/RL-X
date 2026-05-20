from rl_x.environments.environment_manager import (
    extract_environment_name_from_file,
    register_environment,
)

from .create_env import create_train_and_eval_env
from .default_config import get_config
from .general_properties import GeneralProperties


try:
    ROBOCUP_SOCCER_FCP_LOCOMOTION_MJX_ENV = extract_environment_name_from_file(__file__)
except IndexError:
    ROBOCUP_SOCCER_FCP_LOCOMOTION_MJX_ENV = (
        "custom_mujoco.robocup_soccer.fcp_locomotion.mjx"
    )

register_environment(
    ROBOCUP_SOCCER_FCP_LOCOMOTION_MJX_ENV,
    get_config,
    create_train_and_eval_env,
    GeneralProperties,
)
