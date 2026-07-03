import numpy as np


class DefaultActionDelay:
    def __init__(self, env):
        self.env = env

        configured_max_nr_delay_steps = env.env_config["domain_randomization"]["action_delay"]["max_nr_delay_steps"]
        self.nr_delay_steps = int(round(0.020 * env.control_frequency_hz))
        self.max_nr_delay_steps = max(configured_max_nr_delay_steps, self.nr_delay_steps)
        self.mixed_chance = env.env_config["domain_randomization"]["action_delay"]["mixed_chance"]

        self.current_mixed = False
        self.current_nr_delay_steps = 0


    def init(self):
        self.env.internal_state["action_current_mixed"] = False
        self.env.internal_state["action_current_nr_delay_steps"] = 0


    def setup(self):
        self.env.internal_state["action_history"] = np.zeros((self.max_nr_delay_steps + 1, self.env.nr_actuator_joints))


    def sample(self):
        self.env.internal_state["action_current_mixed"] = self.env.np_rng.uniform() < self.mixed_chance
        self.env.internal_state["action_current_nr_delay_steps"] = 0


    def delay_action(self, action):
        self.env.internal_state["action_history"] = np.roll(self.env.internal_state["action_history"], -1, axis=0)
        self.env.internal_state["action_history"][-1] = action.copy()

        chosen_action = self.env.internal_state["action_history"][-1 - self.nr_delay_steps]

        return chosen_action
