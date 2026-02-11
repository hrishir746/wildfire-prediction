import numpy as np
import gym
from gym import spaces
from src.simulation.rothermel_simulator import rothermel_fire_spread

class WildfireRLEnv(gym.Env):
    """
    RL environment for wildfire spread prediction.
    Observation: (input_stack, current_burn_mask)
    Action: adjustment to spread parameters or per-tile ignition
    Reward: increase in IoU with real fire perimeter
    """
    def __init__(self, input_stack, real_perimeter, max_steps=20, map_size=(512, 512)):
        """
        input_stack: (C, H, W) or (batch, C, H, W)
        real_perimeter: (H, W) or (batch, H, W)
        map_size: tuple, e.g. (512, 512) for larger maps
        """
        super().__init__()
        # If input is not already the right size, resize
        from skimage.transform import resize
        if input_stack.shape[-2:] != map_size:
            input_stack = np.array([
                resize(inp, map_size, order=1, preserve_range=True, anti_aliasing=True)
                for inp in input_stack
            ]) if input_stack.ndim == 3 else np.array([
                resize(inp, map_size, order=1, preserve_range=True, anti_aliasing=True)
                for inp in input_stack
            ])
        if real_perimeter.shape[-2:] != map_size:
            real_perimeter = resize(real_perimeter, map_size, order=0, preserve_range=True, anti_aliasing=False)
        self.input_stack = input_stack  # shape: (C, H, W)
        self.real_perimeter = real_perimeter  # shape: (H, W), binary mask
        self.max_steps = max_steps
        self.H, self.W = map_size
        self.action_space = spaces.Box(low=0, high=1, shape=(self.H, self.W), dtype=np.float32)  # ignition prob
        self.observation_space = spaces.Box(low=0, high=1, shape=(input_stack.shape[0]+1, self.H, self.W), dtype=np.float32)
        self.reset()

    def reset(self):
        self.burn_mask = np.zeros((self.H, self.W), dtype=np.float32)
        self.steps = 0
        self.last_iou = 0.0
        return self._get_obs()

    def _get_obs(self):
        # Stack input + current burn mask
        return np.concatenate([self.input_stack, self.burn_mask[None]], axis=0)

    def step(self, action):
        # Action: ignition probability per pixel (0-1)
        ignition = (action > 0.5).astype(np.float32)
        # Simulate one step of fire spread from new ignitions
        new_burn = rothermel_fire_spread(
            elevation=self.input_stack[0],
            fuel_load=self.input_stack[3],
            wind_speed=self.input_stack[6],
            ignition_row=None, ignition_col=None,
            duration_hours=1.0, integration_steps=1,
            ignition_mask=ignition
        )
        self.burn_mask = np.clip(self.burn_mask + new_burn, 0, 1)
        self.steps += 1
        # Reward: IoU improvement
        intersection = (self.burn_mask * self.real_perimeter).sum()
        union = ((self.burn_mask + self.real_perimeter) > 0).sum()
        iou = intersection / (union + 1e-6)
        reward = iou - self.last_iou
        self.last_iou = iou
        done = self.steps >= self.max_steps or iou > 0.95
        info = {'iou': iou, 'map_size': (self.H, self.W)}
        return self._get_obs(), reward, done, info
