from rl_x.algorithms.reppo.flax_full_jit.base import RePPOBase
from rl_x.algorithms.reppo.flax_full_jit.general_properties import GeneralProperties


class RePPO(RePPOBase):
    def general_properties():
        return GeneralProperties
