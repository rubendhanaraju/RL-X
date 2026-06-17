from rl_x.environments.environment_manager import extract_environment_name_from_file, register_environment
from rl_x.environments.custom_mujoco.robocup_soccer.dribbling_residual.mjx.create_env import create_train_and_eval_env
from rl_x.environments.custom_mujoco.robocup_soccer.dribbling_residual.mjx.default_config import get_config
from rl_x.environments.custom_mujoco.robocup_soccer.dribbling_residual.mjx.general_properties import GeneralProperties


ROBOCUP_SOCCER_DRIBBLING_RESIDUAL_MJX_ENV = extract_environment_name_from_file(__file__)
register_environment(
    ROBOCUP_SOCCER_DRIBBLING_RESIDUAL_MJX_ENV,
    get_config,
    create_train_and_eval_env,
    GeneralProperties,
)
