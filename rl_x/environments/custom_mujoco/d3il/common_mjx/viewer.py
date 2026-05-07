import mujoco
import mujoco.viewer


class MujocoViewer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.handle = mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.camera = self.handle.cam

    def is_running(self):
        return self.handle.is_running()

    def render(self, data):
        if not self.is_running():
            return False

        with self.handle.lock():
            mujoco.mj_copyData(self.data, self.model, data)
        self.handle.sync()
        return self.is_running()

    def close(self):
        self.handle.close()
