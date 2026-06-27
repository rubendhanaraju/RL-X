class StepProbabilitySampling:
    def __init__(self, env, probability=0.002):
        self.env = env
        self.probability = probability


    def setup(self, curriculum_coeff=None):
        return False


    def step(self, curriculum_coeff=None):
        if curriculum_coeff is None:
            curriculum_coeff = self.env.internal_state["env_curriculum_coeff"]
        return self.env.np_rng.uniform() < self.probability * curriculum_coeff
