from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from .default_config import get_config
from .general_properties import GeneralProperties
from .tr_vbd_moe import TRVBDMoE


TR_VBD_MOE_FLAX_FULL_JIT = extract_algorithm_name_from_file(__file__)
register_algorithm(TR_VBD_MOE_FLAX_FULL_JIT, get_config, TRVBDMoE, GeneralProperties)
