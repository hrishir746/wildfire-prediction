"""
Rothermel-inspired fire spread simulator.

Implements a 2D cellular automaton fire spread model based on:
- Rothermel (1972), USDA Forest Service Research Paper INT-115: rate of spread (R)
  and reaction intensity from wind, slope, and fuel; wind and slope coefficients
  (spread favored downwind and uphill). Our weights approximate this structure.
- FlamMap (USFS): uses Rothermel (1972) surface fire spread; terrain, fuels, weather
  as inputs. This simulator is a simplified 2D implementation for baseline comparison.

Fire spreads to 8 neighbors with weights from wind direction/speed, slope/aspect,
and fuel density. Scope: surface fire, quasi-steady spread (no crown fire or spotting).
See SOURCES.md and Rothermel (1972) for limitations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


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


class FireSpreadSimulator:
    """
    Deterministic fire spread simulation using Rothermel-inspired rules.

    Fire state: 0 = unburned, 1 = burning, 2 = burned.
    Spread to 8 neighbors is weighted by wind direction and slope (uphill favors spread).
    """

    # 8-neighbor offsets (row, col) - N, NE, E, SE, S, SW, W, NW
    NEIGHBOR_OFFSETS = np.array([
        (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)
    ])

    # Direction angles (degrees from North, clockwise) for each neighbor
    DIRECTION_ANGLES = np.array([0, 45, 90, 135, 180, 225, 270, 315])

    def __init__(self, params: Optional[SimulationParams] = None):
        self.params = params or SimulationParams()

    def _compute_spread_weights(
        self,
        wind_speed: float,
        wind_direction_deg: float,
        slope_deg: np.ndarray,
        aspect_deg: np.ndarray,
        fuel_density: np.ndarray,
    ) -> np.ndarray:
        """
        Compute spread probability/weight from each cell to each of 8 neighbors.

        Rothermel: ROS increases with wind (in wind direction) and slope (uphill).
        Returns shape (rows, cols, 8) for 8 neighbor directions.
        """
        rows, cols = slope_deg.shape
        weights = np.ones((rows, cols, 8), dtype=np.float32)

        # Wind effect: spread favored in wind direction
        # wind_direction_deg = direction wind is blowing FROM (meteorological convention)
        # Fire spreads downwind, so we want spread favored in (wind_dir + 180) % 360
        fire_spread_direction = (wind_direction_deg + 180) % 360

        for k, angle in enumerate(self.DIRECTION_ANGLES):
            angle_diff = np.abs((angle - fire_spread_direction + 180) % 360 - 180)
            wind_component = np.cos(np.radians(angle_diff))
            weights[:, :, k] *= 1.0 + self.params.wind_factor * wind_speed * np.maximum(0, wind_component)

        # Slope effect: uphill spread is faster (Rothermel)
        # Aspect = direction slope faces; fire spreads faster uphill (toward aspect)
        slope_rad = np.radians(slope_deg)
        aspect_rad = np.radians(aspect_deg)

        for k, (dr, dc) in enumerate(self.NEIGHBOR_OFFSETS):
            # Direction to neighbor (from center)
            neighbor_angle = np.degrees(np.arctan2(dc, -dr)) % 360
            # Uphill = toward aspect; downhill = away
            angle_to_aspect = np.abs((neighbor_angle - aspect_deg + 180) % 360 - 180)
            slope_component = np.cos(np.radians(angle_to_aspect))
            slope_effect = 1.0 + self.params.slope_factor * np.tan(slope_rad) * np.maximum(0, slope_component)
            weights[:, :, k] *= np.clip(slope_effect, 0.1, 5.0)

        # Fuel density modulates spread (simplified)
        fuel_factor = 0.5 + 0.5 * np.clip(fuel_density, 0, 1)
        weights *= fuel_factor[:, :, np.newaxis]

        return weights

    def step(
        self,
        fire_state: np.ndarray,
        slope_deg: np.ndarray,
        aspect_deg: np.ndarray,
        wind_speed: float,
        wind_direction_deg: float,
        fuel_density: np.ndarray,
    ) -> np.ndarray:
        """
        Advance fire by one time step.

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
        new_fire = np.zeros_like(fire_state, dtype=bool)

        for k, (dr, dc) in enumerate(self.NEIGHBOR_OFFSETS):
            # Roll burning mask: rolled_burning[i,j] = burning at neighbor (i+dr, j+dc)
            rolled_burning = np.roll(burning, (-dc, -dr), axis=(1, 0))
            # Weight from neighbor toward us: roll so w[i,j] = weight at neighbor
            opp_k = (k + 4) % 8
            w = np.roll(weights[:, :, opp_k], (dr, dc), axis=(0, 1))
            # Probability of ignition: weight * burning_neighbor
            ignition_prob = rolled_burning.astype(float) * w
            # Threshold: convert to binary ignition (simplified)
            threshold = 0.5
            new_fire |= (ignition_prob > threshold) & (fire_state == 0)

        # Update state: new ignitions become burning
        fire_state = fire_state.copy()
        fire_state[new_fire] = 1
        fire_state[burning] = 2  # previously burning now burned

        return fire_state

    def run(
        self,
        ignition_mask: np.ndarray,
        slope_deg: np.ndarray,
        aspect_deg: np.ndarray,
        wind_speed: float = 5.0,
        wind_direction_deg: float = 270.0,
        fuel_density: Optional[np.ndarray] = None,
        max_steps: int = 100,
    ) -> list[np.ndarray]:
        """
        Run simulation from ignition to max_steps.

        Args:
            ignition_mask: (rows, cols) bool, True = initial ignition
            slope_deg, aspect_deg: terrain
            wind_speed, wind_direction_deg: weather
            fuel_density: optional, defaults to uniform 1.0
            max_steps: maximum time steps

        Returns:
            List of fire_state arrays (one per step, including t=0)
        """
        rows, cols = ignition_mask.shape
        if fuel_density is None:
            fuel_density = np.ones((rows, cols), dtype=np.float32)

        fire_state = np.zeros((rows, cols), dtype=np.int32)
        fire_state[ignition_mask] = 1

        history = [fire_state.copy()]
        for _ in range(max_steps - 1):
            if not np.any(fire_state == 1):
                break
            fire_state = self.step(
                fire_state, slope_deg, aspect_deg,
                wind_speed, wind_direction_deg, fuel_density
            )
            history.append(fire_state.copy())

        return history

    def create_demo_terrain(
        self,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create synthetic terrain for demonstration (no real data required).

        Returns slope_deg, aspect_deg, fuel_density.
        """
        rng = np.random.default_rng(seed)
        rows = rows or self.params.grid_rows
        cols = cols or self.params.grid_cols

        # Simple gradient + noise (much smaller scale to avoid extreme slopes)
        y = np.linspace(0, 1, rows)
        x = np.linspace(0, 1, cols)
        xx, yy = np.meshgrid(x, y)
        elev = 0.05 * yy + 0.03 * np.sin(4 * xx) + 0.02 * rng.random((rows, cols))
        # Don't cumsum - just use the values directly for gradients

        # Gradient for slope/aspect
        dy, dx = np.gradient(elev, 1.0 / rows, 1.0 / cols)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.clip(np.degrees(slope_rad), 0, 45)  # Cap at 45°
        aspect_deg = (np.degrees(np.arctan2(-dx, dy)) + 360) % 360

        # Fuel density: vary by terrain
        fuel_density = 0.5 + 0.5 * (1 - slope_deg / 45)
        fuel_density = np.clip(fuel_density + 0.1 * rng.random((rows, cols)), 0.2, 1.0)

        return slope_deg.astype(np.float32), aspect_deg.astype(np.float32), fuel_density.astype(np.float32)


def rothermel_fire_spread(
    elevation: np.ndarray,
    fuel_load: np.ndarray,
    wind_speed: float,
    ignition_row: int,
    ignition_col: int,
    duration_hours: float = 12.0,
    integration_steps: int = 48,
    wind_direction_deg: float = 270.0,
) -> np.ndarray:
    """
    Convenience function for running fire spread simulation.
    
    This is a simplified wrapper around FireSpreadSimulator for backward compatibility
    with scripts that use the functional interface.
    
    Args:
        elevation: 2D array of elevation values (meters)
        fuel_load: 2D array of fuel density [0, 1]
        wind_speed: Wind speed (m/s)
        ignition_row: Row index of ignition point
        ignition_col: Column index of ignition point
        duration_hours: Simulation duration in hours
        integration_steps: Number of timesteps to simulate
        wind_direction_deg: Wind direction in degrees (meteorological convention)
    
    Returns:
        2D binary array of burned area (0=unburned, 1=burned)
    """
    rows, cols = elevation.shape
    
    # Compute slope and aspect from elevation
    dy, dx = np.gradient(elevation, 1.0)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    aspect_deg = (np.degrees(np.arctan2(-dx, dy)) + 360) % 360
    
    # Create ignition mask
    ignition_mask = np.zeros((rows, cols), dtype=bool)
    ignition_mask[ignition_row, ignition_col] = True
    
    # Initialize simulator
    params = SimulationParams(
        grid_rows=rows,
        grid_cols=cols,
        cell_size_m=30.0,
        time_step_s=duration_hours * 3600 / integration_steps
    )
    simulator = FireSpreadSimulator(params)
    
    # Run simulation
    history = simulator.run(
        ignition_mask=ignition_mask,
        slope_deg=slope_deg,
        aspect_deg=aspect_deg,
        wind_speed=wind_speed,
        wind_direction_deg=wind_direction_deg,
        fuel_density=fuel_load,
        max_steps=integration_steps
    )
    
    # Return final burned area as binary mask
    final_state = history[-1]
    return (final_state > 0).astype(np.float32)
