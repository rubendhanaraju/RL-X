class StepProbabilitySampling:
    def __init__(self, env, probability=0.002):
        self.env = env
        self.probability = probability


    def setup(self):
        return False


    def step(self):
        probability = self.env.internal_state.get("command_resampling_probability", self.probability)
        return self.env.np_rng.uniform() < probability
