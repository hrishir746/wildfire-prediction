"""
GPU-accelerated Rothermel-inspired fire spread simulator using PyTorch.

This is a PyTorch implementation of the Rothermel simulator that runs on GPU,
providing significant speedup for batch data generation.

All NumPy operations are replaced with PyTorch equivalents for GPU execution.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Union


@dataclass
class SimulationParams:
    """Parameters for Rothermel-inspired fire spread."""
    grid_rows: int = 128
    grid_cols: int = 128
    cell_size_m: float = 30.0
    time_step_s: float = 60.0
    base_ros: float = 0.5         # base rate of spread (m/min)
    wind_factor: float = 0.4      # wind influence on ROS
    slope_factor: float = 0.3     # slope influence on ROS


class FireSpreadSimulatorGPU:
    """
    GPU-accelerated fire spread simulation using PyTorch.
    
    Fire state: 0 = unburned, 1 = burning, 2 = burned.
    Spread to 8 neighbors is weighted by wind direction and slope (uphill favors spread).
    """

    # 8-neighbor offsets (row, col) - N, NE, E, SE, S, SW, W, NW
    NEIGHBOR_OFFSETS = [
        (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)
    ]

    # Direction angles (degrees from North, clockwise) for each neighbor
    DIRECTION_ANGLES = torch.tensor([0, 45, 90, 135, 180, 225, 270, 315], dtype=torch.float32)

    def __init__(self, params: Optional[SimulationParams] = None, device: Optional[torch.device] = None):
        self.params = params or SimulationParams()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.direction_angles = self.DIRECTION_ANGLES.to(self.device)

    def _compute_spread_weights(
        self,
        wind_speed: float,
        wind_direction_deg: float,
        slope_deg: torch.Tensor,
        aspect_deg: torch.Tensor,
        fuel_density: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute spread probability/weight from each cell to each of 8 neighbors.

        Returns shape (rows, cols, 8) for 8 neighbor directions.
        """
        rows, cols = slope_deg.shape
        weights = torch.ones((rows, cols, 8), dtype=torch.float32, device=self.device)

        # Wind effect: spread favored in wind direction
        fire_spread_direction = (wind_direction_deg + 180) % 360

        for k, angle in enumerate(self.DIRECTION_ANGLES):
            angle_diff = torch.abs((angle - fire_spread_direction + 180) % 360 - 180)
            wind_component = torch.cos(torch.deg2rad(angle_diff))
            weights[:, :, k] *= 1.0 + self.params.wind_factor * wind_speed * torch.maximum(
                torch.tensor(0.0, device=self.device), wind_component
            )

        # Slope effect: uphill spread is faster
        slope_rad = torch.deg2rad(slope_deg)
        aspect_rad = torch.deg2rad(aspect_deg)

        for k, (dr, dc) in enumerate(self.NEIGHBOR_OFFSETS):
            # Direction to neighbor
            neighbor_angle = (torch.atan2(
                torch.tensor(dc, dtype=torch.float32, device=self.device),
                torch.tensor(-dr, dtype=torch.float32, device=self.device)
            ) * 180 / torch.pi) % 360
            
            angle_to_aspect = torch.abs((neighbor_angle - aspect_deg + 180) % 360 - 180)
            slope_component = torch.cos(torch.deg2rad(angle_to_aspect))
            slope_effect = 1.0 + self.params.slope_factor * torch.tan(slope_rad) * torch.maximum(
                torch.tensor(0.0, device=self.device), slope_component
            )
            weights[:, :, k] *= torch.clamp(slope_effect, 0.1, 5.0)

        # Fuel density modulates spread
        fuel_factor = 0.5 + 0.5 * torch.clamp(fuel_density, 0, 1)
        weights *= fuel_factor.unsqueeze(-1)

        return weights

    def step(
        self,
        fire_state: torch.Tensor,
        slope_deg: torch.Tensor,
        aspect_deg: torch.Tensor,
        wind_speed: float,
        wind_direction_deg: float,
        fuel_density: torch.Tensor,
    ) -> torch.Tensor:
        """
        Advance fire by one time step on GPU.

        Args:
            fire_state: (rows, cols) with 0=unburned, 1=burning, 2=burned
            slope_deg: slope in degrees
            aspect_deg: aspect in degrees (0=N, 90=E)
            wind_speed: m/s
            wind_direction_deg: degrees (meteorological: from where wind blows)
            fuel_density: normalized [0,1]

        Returns:
            Updated fire_state
        """
        rows, cols = fire_state.shape
        weights = self._compute_spread_weights(
            wind_speed, wind_direction_deg, slope_deg, aspect_deg, fuel_density
        )

        # Cells that can ignite neighbors (burning)
        burning = (fire_state == 1)
        new_fire = torch.zeros_like(fire_state, dtype=torch.bool)

        for k, (dr, dc) in enumerate(self.NEIGHBOR_OFFSETS):
            # Roll burning mask
            rolled_burning = torch.roll(burning, shifts=(-dc, -dr), dims=(1, 0))
            # Weight from neighbor toward us
            opp_k = (k + 4) % 8
            w = torch.roll(weights[:, :, opp_k], shifts=(dr, dc), dims=(0, 1))
            # Probability of ignition
            ignition_prob = rolled_burning.float() * w
            # Threshold
            threshold = 0.5
            new_fire |= (ignition_prob > threshold) & (fire_state == 0)

        # Update state
        fire_state = fire_state.clone()
        fire_state[new_fire] = 1
        fire_state[burning] = 2

        return fire_state

    def run(
        self,
        ignition_mask: torch.Tensor,
        slope_deg: torch.Tensor,
        aspect_deg: torch.Tensor,
        wind_speed: float = 5.0,
        wind_direction_deg: float = 270.0,
        fuel_density: Optional[torch.Tensor] = None,
        max_steps: int = 100,
    ) -> list[torch.Tensor]:
        """
        Run simulation from ignition to max_steps on GPU.

        Returns:
            List of fire_state tensors (one per step, including t=0)
        """
        rows, cols = ignition_mask.shape
        if fuel_density is None:
            fuel_density = torch.ones((rows, cols), dtype=torch.float32, device=self.device)

        fire_state = torch.zeros((rows, cols), dtype=torch.int32, device=self.device)
        fire_state[ignition_mask] = 1

        history = [fire_state.clone()]
        for _ in range(max_steps - 1):
            if not torch.any(fire_state == 1):
                break
            fire_state = self.step(
                fire_state, slope_deg, aspect_deg,
                wind_speed, wind_direction_deg, fuel_density
            )
            history.append(fire_state.clone())

        return history

    def create_demo_terrain(
        self,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create synthetic terrain on GPU for demonstration.

        Returns slope_deg, aspect_deg, fuel_density as PyTorch tensors on device.
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        rows = rows or self.params.grid_rows
        cols = cols or self.params.grid_cols

        # Simple gradient + noise
        y = torch.linspace(0, 1, rows, device=self.device)
        x = torch.linspace(0, 1, cols, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        yy = yy.T
        xx = xx.T
        
        elev = 0.05 * yy + 0.03 * torch.sin(4 * xx) + 0.02 * torch.rand((rows, cols), device=self.device)

        # Compute gradient for slope/aspect
        # PyTorch doesn't have np.gradient, so we use convolution-based approach
        dy = torch.zeros_like(elev)
        dx = torch.zeros_like(elev)
        
        # Central differences (interior points)
        dy[1:-1, :] = (elev[2:, :] - elev[:-2, :]) / (2.0 / rows)
        dx[:, 1:-1] = (elev[:, 2:] - elev[:, :-2]) / (2.0 / cols)
        
        # Forward/backward differences at edges
        dy[0, :] = (elev[1, :] - elev[0, :]) / (1.0 / rows)
        dy[-1, :] = (elev[-1, :] - elev[-2, :]) / (1.0 / rows)
        dx[:, 0] = (elev[:, 1] - elev[:, 0]) / (1.0 / cols)
        dx[:, -1] = (elev[:, -1] - elev[:, -2]) / (1.0 / cols)

        slope_rad = torch.atan(torch.sqrt(dx**2 + dy**2))
        slope_deg = torch.clamp(torch.rad2deg(slope_rad), 0, 45)
        aspect_deg = (torch.rad2deg(torch.atan2(-dx, dy)) + 360) % 360

        # Fuel density
        fuel_density = 0.5 + 0.5 * (1 - slope_deg / 45)
        fuel_density = torch.clamp(fuel_density + 0.1 * torch.rand((rows, cols), device=self.device), 0.2, 1.0)

        return slope_deg, aspect_deg, fuel_density

    def create_demo_terrain_batch(
        self,
        batch_size: int,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        base_seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create a batch of synthetic terrains on GPU for faster data generation.

        Returns:
            slope_deg: (batch_size, rows, cols)
            aspect_deg: (batch_size, rows, cols)
            fuel_density: (batch_size, rows, cols)
        """
        rows = rows or self.params.grid_rows
        cols = cols or self.params.grid_cols

        slopes = []
        aspects = []
        fuels = []

        for i in range(batch_size):
            seed = (base_seed + i) if base_seed is not None else None
            slope, aspect, fuel = self.create_demo_terrain(rows, cols, seed)
            slopes.append(slope)
            aspects.append(aspect)
            fuels.append(fuel)

        return (
            torch.stack(slopes),
            torch.stack(aspects),
            torch.stack(fuels)
        )
