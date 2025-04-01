from torchrl.envs import GymEnv, TransformedEnv, RewardSum

from florl.task.abc import OnlineTask


class GymTask(OnlineTask):
    def __init__(self, env_name: str):
        super().__init__()
        self._env_name = env_name

    def create_env(self, mode):
        env = GymEnv(env_name=self._env_name)
        if mode in ["train", "evaluate"]:
            env = TransformedEnv(
                env, RewardSum(in_keys=env.reward_keys, out_keys=["episode_reward"])
            )
        return env
