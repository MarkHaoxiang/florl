from torchrl.envs import GymEnv, TransformedEnv, RewardSum

from florl.task.abc import Benchmark
from florl.task.online import OnlineTask


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


MuJoCo = Benchmark(
    [
        GymTask(env_name)
        for env_name in [
            "HalfCheetah-v4",
            "Hopper-v4",
            "Walker2d-v4",
            "Ant-v4",
            "Reacher-v4",
            "Swimmer-v4",
            "InvertedPendulum-v4",
            "InvertedDoublePendulum-v4",
        ]
    ]
)

ClassicalControlContinuous = Benchmark(
    [
        GymTask(env_name)
        for env_name in [
            "MountainCarContinuous-v0",
            "Pendulum-v1",
        ]
    ]
)

ClassicalControlDiscrete = Benchmark(
    [
        GymTask(env_name)
        for env_name in [
            "CartPole-v1",
            "Acrobot-v1",
            "MountainCar-v0",
            "Pendulum-v1",
        ]
    ]
)
