import gym
from typing import Callable
import os
import sys
from gym import logger
from isaacgymenvs.tasks.base.vec_task import VecTask
from pyffmpeg import FFmpeg
import shutil


def capped_cubic_video_schedule(episode_id: int) -> bool:
    """The default episode trigger.

    This function will trigger recordings at the episode indices 0, 1, 4, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Args:
        episode_id: The episode number

    Returns:
        If to apply a video schedule number
    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


class RecordWrapper(gym.Wrapper):

    def __init__(self, env: VecTask,
                 video_folder: str,
                 episode_trigger: Callable[[int], bool] = None,
                 step_trigger: Callable[[int], bool] = None,
                 video_length: int = 0,
                 max_length: int = 0,
                 name_prefix: str = "rl-video", ):

        super().__init__(env)

        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum(x is not None for x in [episode_trigger, step_trigger])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_recorder = None

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.episode_id = 0

        self.max_length = max_length

    def reset(self, **kwargs):
        """Reset the environment using kwargs and then starts recording if video enabled."""
        observations = super().reset(**kwargs)
        if not self.recording and self._video_enabled():
            self.start_video_recorder()
        return observations

    def start_video_recorder(self):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        # self.video_recorder = video_recorder.VideoRecorder(
        #     env=self.env,
        #     base_path=base_path,
        #     metadata={"step_id": self.step_id, "episode_id": self.episode_id},
        # )
        self.video_recorder = VideoRecorder(self.env, base_path)

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        else:
            return self.episode_trigger(self.episode_id)

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        observations, rewards, dones, infos = super().step(action)

        # increment steps and episodes
        self.step_id += 1
        if not self.is_vector_env:
            if dones:
                self.episode_id += 1
        elif dones[0]:
            self.episode_id += 1

        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames >= self.video_length:
                    self.close_video_recorder()
            else:
                if not self.is_vector_env:
                    if dones:
                        self.close_video_recorder()
                elif dones[0]:
                    self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()

        if self.max_length > 0 and self.step_id >= self.max_length:
            self.close_video_recorder()
            sys.exit()

        return observations, rewards, dones, infos

    def _step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        observations, rewards, dones, infos = super().step(action)

        # increment steps and episodes
        self.step_id += 1
        if not self.is_vector_env:
            if dones:
                self.episode_id += 1
        elif dones[0]:
            self.episode_id += 1

        if self.recording:
            if self.video_length > 0:
                if self.recorded_frames >= self.video_length:
                    self.close_video_recorder()
            else:
                if not self.is_vector_env:
                    if dones:
                        self.close_video_recorder()
                elif dones[0]:
                    self.close_video_recorder()

        if self.max_length > 0 and self.step_id >= self.max_length:
            self.close_video_recorder()
            sys.exit()

        return observations, rewards, dones, infos

    def _render(self):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        super().render()

        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1

        elif self._video_enabled():
            self.start_video_recorder()

    def close_video_recorder(self):
        """Closes the video recorder if currently recording."""
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        self.close_video_recorder()

    def __del__(self):
        """Closes the video recorder."""
        self.close_video_recorder()


class VideoRecorder():
    def __init__(self, env: VecTask, base_path) -> None:
        self.env = env
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

        self.frame_idx = -1

    def capture_frame(self):
        self.frame_idx += 1
        file_name = os.path.join(self.base_path, str(self.frame_idx) + '.png')
        self.env.gym.write_viewer_image_to_file(self.env.viewer, file_name)

    def close(self, delete_imgs=True):
        self.frame_idx = -1
        ff = FFmpeg()
        option = "-framerate 30 -start_number 0 -i " + self.base_path + r"/%d.png -pix_fmt yuv420p " + self.base_path + ".mp4"
        print("ffmpeg option:", option)
        ff.options(option)
        print("Export video: " + self.base_path + ".mp4")

        if delete_imgs:
            shutil.rmtree(self.base_path, ignore_errors=True)
            print("Delete '%s' directory" % self.base_path)
