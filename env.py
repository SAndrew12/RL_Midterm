import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
import math


@dataclass
class AircraftState:
    """Represents the state of a single aircraft."""
    x: float  # Nautical miles from origin
    y: float  # Nautical miles from origin
    altitude: float  # Feet
    heading: float  # Degrees (0-360)
    speed: float  # Knots
    vertical_speed: float = 0.0  # Feet per minute
    callsign: str = "FLIGHT1"

    def copy(self) -> 'AircraftState':
        return AircraftState(
            x=self.x, y=self.y, altitude=self.altitude,
            heading=self.heading, speed=self.speed,
            vertical_speed=self.vertical_speed, callsign=self.callsign
        )


@dataclass
class AircraftPerformance:
    """Aircraft performance parameters."""
    min_speed: float = 150.0  # Knots
    max_speed: float = 250.0  # Knots
    standard_turn_rate: float = 3.0  # Degrees per second
    max_climb_rate: float = 2000.0  # Feet per minute
    max_descent_rate: float = 2000.0  # Feet per minute
    acceleration: float = 5.0  # Knots per second


@dataclass
class Runway:
    """Runway configuration with Final Approach Fix (FAF)."""
    x: float  # FAF x position (nautical miles)
    y: float  # FAF y position (nautical miles)
    heading: float  # Runway heading (degrees)
    faf_altitude: float = 3000.0  # Required altitude at FAF (feet)
    glideslope_angle: float = 3.0  # Degrees
    localizer_width: float = 45.0  # Degrees (each side)


@dataclass
class Airspace:
    """Defines the controlled airspace sector."""
    x_min: float = -50.0  # Nautical miles
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0
    floor: float = 0.0  # Feet
    ceiling: float = 15000.0  # Feet


@dataclass
class MVA:
    """Minimum Vectoring Altitude for a region."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    altitude: float  # Minimum altitude in feet


class ATCGymEnv(gym.Env):
    """
    Air Traffic Control Gymnasium Environment.

    The agent controls aircraft by issuing heading and altitude commands.
    The goal is to guide aircraft to the Final Approach Fix (FAF) at the
    correct altitude and heading.

    Observation Space:
        - Aircraft position (x, y) relative to FAF
        - Aircraft altitude
        - Aircraft heading
        - Aircraft speed
        - Distance to FAF
        - Bearing to FAF
        - Heading difference to runway

    Action Space (Continuous):
        - Heading change (-180 to 180 degrees)
        - Target altitude change (-5000 to 5000 feet)

    Action Space (Discrete, optional):
        - 9 actions: combinations of turn left/right/straight and climb/descend/level

    Rewards:
        - Positive reward for reaching FAF at correct altitude and heading
        - Negative reward for each timestep (encourages efficiency)
        - Heavy penalty for leaving airspace
        - Heavy penalty for descending below MVA
        - Heavy penalty for loss of separation (multi-aircraft)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
            self,
            render_mode: Optional[str] = None,
            continuous_actions: bool = True,
            num_aircraft: int = 1,
            max_steps: int = 500,
            timestep: float = 5.0,  # seconds per step
            random_spawn: bool = True,
            difficulty: str = "medium",  # easy, medium, hard
    ):
        super().__init__()

        self.render_mode = render_mode
        self.continuous_actions = continuous_actions
        self.num_aircraft = num_aircraft
        self.max_steps = max_steps
        self.timestep = timestep
        self.random_spawn = random_spawn
        self.difficulty = difficulty

        # Initialize environment components
        self._setup_environment()

        # Define action space
        if self.continuous_actions:
            # [heading_change, altitude_change]
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0]),  # Normalized
                high=np.array([1.0, 1.0]),
                dtype=np.float32
            )
        else:
            # 9 discrete actions: 3 heading options x 3 altitude options
            self.action_space = spaces.Discrete(9)

        # Define observation space
        # For each aircraft: [x, y, altitude, heading, speed, dist_to_faf,
        #                     bearing_to_faf, heading_diff_to_runway, vertical_speed]
        obs_dim = 9 * self.num_aircraft
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # State variables
        self.aircraft: List[AircraftState] = []
        self.current_step = 0
        self.done = False
        self.truncated = False

        # Rendering
        self.screen = None
        self.clock = None
        self.screen_width = 800
        self.screen_height = 800

    def _setup_environment(self):
        """Set up the airspace, runway, and MVAs based on difficulty."""
        # Runway with FAF
        self.runway = Runway(x=0.0, y=0.0, heading=270.0, faf_altitude=3000.0)

        # Airspace boundaries
        self.airspace = Airspace()

        # Minimum Vectoring Altitudes (terrain)
        self.mvas = self._create_mvas()

        # Aircraft performance
        self.performance = AircraftPerformance()

        # Success/failure thresholds
        self.faf_radius = 1.0  # NM - distance to FAF for capture
        self.altitude_tolerance = 300  # Feet
        self.heading_tolerance = 45.0  # Degrees
        self.min_separation_horizontal = 3.0  # NM
        self.min_separation_vertical = 1000  # Feet

        # Reward weights
        self.reward_weights = {
            'step_penalty': -0.1,
            'distance_reward': 0.01,
            'altitude_reward': 0.01,
            'heading_reward': 0.01,
            'faf_capture': 100.0,
            'mva_violation': -50.0,
            'airspace_exit': -100.0,
            'separation_loss': -100.0,
        }

    def _create_mvas(self) -> List[MVA]:
        """Create minimum vectoring altitude regions (simulated terrain)."""
        mvas = []

        if self.difficulty == "easy":
            # Single low MVA everywhere
            mvas.append(MVA(-50, 50, -50, 50, 2000))
        elif self.difficulty == "medium":
            # Some higher terrain areas
            mvas.append(MVA(-50, 50, -50, 50, 2000))  # Base
            mvas.append(MVA(10, 30, 10, 30, 4000))  # Mountain
            mvas.append(MVA(-30, -10, -30, -10, 3500))  # Hills
        else:  # hard
            # Complex terrain
            mvas.append(MVA(-50, 50, -50, 50, 2000))
            mvas.append(MVA(5, 35, 5, 35, 5000))
            mvas.append(MVA(-35, -5, -35, -5, 4500))
            mvas.append(MVA(-20, 20, 20, 40, 4000))
            mvas.append(MVA(-40, -20, 0, 20, 3500))

        return mvas

    def _get_mva_at_position(self, x: float, y: float) -> float:
        """Get the minimum vectoring altitude at a given position."""
        max_mva = 0.0
        for mva in self.mvas:
            if mva.x_min <= x <= mva.x_max and mva.y_min <= y <= mva.y_max:
                max_mva = max(max_mva, mva.altitude)
        return max_mva

    def _spawn_aircraft(self) -> AircraftState:
        """Spawn an aircraft at a random or fixed position."""
        if self.random_spawn:
            # Random spawn at sector boundary
            side = self.np_random.choice(['north', 'south', 'east', 'west'])

            if side == 'north':
                x = self.np_random.uniform(-40, 40)
                y = 45
                heading = self.np_random.uniform(180, 270)
            elif side == 'south':
                x = self.np_random.uniform(-40, 40)
                y = -45
                heading = self.np_random.uniform(0, 90)
            elif side == 'east':
                x = 45
                y = self.np_random.uniform(-40, 40)
                heading = self.np_random.uniform(225, 315)
            else:  # west
                x = -45
                y = self.np_random.uniform(-40, 40)
                heading = self.np_random.uniform(45, 135)

            altitude = self.np_random.uniform(8000, 12000)
            speed = self.np_random.uniform(200, 250)
        else:
            # Fixed spawn for reproducibility
            x = 30.0
            y = 30.0
            altitude = 10000.0
            heading = 225.0
            speed = 220.0

        return AircraftState(
            x=x, y=y, altitude=altitude,
            heading=heading, speed=speed,
            callsign=f"FLIGHT{len(self.aircraft) + 1}"
        )

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Clear and respawn aircraft
        self.aircraft = []
        for _ in range(self.num_aircraft):
            self.aircraft.append(self._spawn_aircraft())

        self.current_step = 0
        self.done = False
        self.truncated = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
            self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep of the environment."""

        # Parse action
        if self.continuous_actions:
            # Scale normalized action to actual values
            heading_change = action[0] * 30.0  # Max 30 degrees per step
            altitude_change = action[1] * 1000.0  # Max 1000 feet per step
        else:
            # Discrete action mapping
            heading_actions = [-20.0, 0.0, 20.0]
            altitude_actions = [-500.0, 0.0, 500.0]

            heading_idx = action // 3
            altitude_idx = action % 3

            heading_change = heading_actions[heading_idx]
            altitude_change = altitude_actions[altitude_idx]

        # Apply action to first aircraft (extend for multi-aircraft)
        self._apply_action(self.aircraft[0], heading_change, altitude_change)

        # Update aircraft positions
        for aircraft in self.aircraft:
            self._update_aircraft(aircraft)

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination conditions
        terminated, truncated = self._check_termination()

        self.current_step += 1

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _apply_action(
            self,
            aircraft: AircraftState,
            heading_change: float,
            altitude_change: float
    ):
        """Apply control action to aircraft."""
        # Calculate turn rate limit
        max_heading_change = self.performance.standard_turn_rate * self.timestep
        heading_change = np.clip(heading_change, -max_heading_change, max_heading_change)

        # Update target heading
        new_heading = (aircraft.heading + heading_change) % 360
        aircraft.heading = new_heading

        # Update vertical speed based on altitude change target
        target_altitude = aircraft.altitude + altitude_change
        target_altitude = np.clip(target_altitude, 0, self.airspace.ceiling)

        if altitude_change > 0:
            aircraft.vertical_speed = min(
                self.performance.max_climb_rate,
                altitude_change * 60 / self.timestep
            )
        elif altitude_change < 0:
            aircraft.vertical_speed = max(
                -self.performance.max_descent_rate,
                altitude_change * 60 / self.timestep
            )
        else:
            aircraft.vertical_speed = 0.0

    def _update_aircraft(self, aircraft: AircraftState):
        """Update aircraft position based on current state."""
        # Convert heading to radians
        heading_rad = math.radians(aircraft.heading)

        # Calculate distance traveled in this timestep
        # Speed is in knots (NM/hour), timestep is in seconds
        distance = aircraft.speed * (self.timestep / 3600)

        # Update position
        aircraft.x += distance * math.sin(heading_rad)
        aircraft.y += distance * math.cos(heading_rad)

        # Update altitude
        altitude_change = aircraft.vertical_speed * (self.timestep / 60)
        aircraft.altitude += altitude_change
        aircraft.altitude = np.clip(aircraft.altitude, 0, self.airspace.ceiling)

    def _calculate_reward(self) -> float:
        """Calculate reward for current state."""
        reward = 0.0
        aircraft = self.aircraft[0]  # Primary aircraft

        # Step penalty (encourages efficiency)
        reward += self.reward_weights['step_penalty']

        # Distance to FAF reward (closer is better)
        dist_to_faf = self._distance_to_faf(aircraft)
        reward += self.reward_weights['distance_reward'] * (1.0 / (dist_to_faf + 0.1))

        # Altitude alignment reward
        altitude_diff = abs(aircraft.altitude - self.runway.faf_altitude)
        if altitude_diff < self.altitude_tolerance:
            reward += self.reward_weights['altitude_reward']

        # Heading alignment reward
        heading_diff = self._heading_difference(
            aircraft.heading,
            self.runway.heading
        )
        if heading_diff < self.heading_tolerance:
            reward += self.reward_weights['heading_reward']

        # Check FAF capture
        if self._check_faf_capture(aircraft):
            reward += self.reward_weights['faf_capture']

        # MVA violation penalty
        mva = self._get_mva_at_position(aircraft.x, aircraft.y)
        if aircraft.altitude < mva:
            reward += self.reward_weights['mva_violation']

        # Airspace exit penalty
        if self._is_outside_airspace(aircraft):
            reward += self.reward_weights['airspace_exit']

        # Separation loss (multi-aircraft)
        if self.num_aircraft > 1:
            if self._check_separation_loss():
                reward += self.reward_weights['separation_loss']

        return reward

    def _distance_to_faf(self, aircraft: AircraftState) -> float:
        """Calculate distance from aircraft to FAF in nautical miles."""
        dx = aircraft.x - self.runway.x
        dy = aircraft.y - self.runway.y
        return math.sqrt(dx * dx + dy * dy)

    def _bearing_to_faf(self, aircraft: AircraftState) -> float:
        """Calculate bearing from aircraft to FAF in degrees."""
        dx = self.runway.x - aircraft.x
        dy = self.runway.y - aircraft.y
        bearing = math.degrees(math.atan2(dx, dy))
        return bearing % 360

    def _heading_difference(self, heading1: float, heading2: float) -> float:
        """Calculate smallest difference between two headings."""
        diff = abs(heading1 - heading2)
        return min(diff, 360 - diff)

    def _check_faf_capture(self, aircraft: AircraftState) -> bool:
        """Check if aircraft has successfully captured the FAF."""
        dist = self._distance_to_faf(aircraft)
        alt_diff = abs(aircraft.altitude - self.runway.faf_altitude)
        hdg_diff = self._heading_difference(aircraft.heading, self.runway.heading)

        return (
                dist < self.faf_radius and
                alt_diff < self.altitude_tolerance and
                hdg_diff < self.heading_tolerance
        )

    def _is_outside_airspace(self, aircraft: AircraftState) -> bool:
        """Check if aircraft has left the controlled airspace."""
        return (
                aircraft.x < self.airspace.x_min or
                aircraft.x > self.airspace.x_max or
                aircraft.y < self.airspace.y_min or
                aircraft.y > self.airspace.y_max
        )

    def _check_separation_loss(self) -> bool:
        """Check for loss of separation between aircraft."""
        for i in range(len(self.aircraft)):
            for j in range(i + 1, len(self.aircraft)):
                ac1, ac2 = self.aircraft[i], self.aircraft[j]

                # Horizontal separation
                dx = ac1.x - ac2.x
                dy = ac1.y - ac2.y
                horiz_dist = math.sqrt(dx * dx + dy * dy)

                # Vertical separation
                vert_dist = abs(ac1.altitude - ac2.altitude)

                if (horiz_dist < self.min_separation_horizontal and
                        vert_dist < self.min_separation_vertical):
                    return True
        return False

    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if episode should terminate."""
        aircraft = self.aircraft[0]

        # Success - FAF captured
        if self._check_faf_capture(aircraft):
            return True, False

        # Failure - outside airspace
        if self._is_outside_airspace(aircraft):
            return True, False

        # Failure - MVA violation
        mva = self._get_mva_at_position(aircraft.x, aircraft.y)
        if aircraft.altitude < mva:
            return True, False

        # Failure - separation loss
        if self.num_aircraft > 1 and self._check_separation_loss():
            return True, False

        # Truncated - max steps reached
        if self.current_step >= self.max_steps:
            return False, True

        return False, False

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        observations = []

        for aircraft in self.aircraft:
            dist_to_faf = self._distance_to_faf(aircraft)
            bearing_to_faf = self._bearing_to_faf(aircraft)
            heading_diff = self._heading_difference(
                aircraft.heading,
                self.runway.heading
            )

            # Normalize observations
            obs = [
                aircraft.x / 50.0,  # Normalize by airspace size
                aircraft.y / 50.0,
                aircraft.altitude / 15000.0,  # Normalize by ceiling
                aircraft.heading / 360.0,
                aircraft.speed / 300.0,
                dist_to_faf / 70.0,  # Max possible distance
                bearing_to_faf / 360.0,
                heading_diff / 180.0,
                aircraft.vertical_speed / 2000.0,
            ]
            observations.extend(obs)

        return np.array(observations, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        aircraft = self.aircraft[0]
        return {
            'distance_to_faf': self._distance_to_faf(aircraft),
            'altitude': aircraft.altitude,
            'heading': aircraft.heading,
            'speed': aircraft.speed,
            'faf_captured': self._check_faf_capture(aircraft),
            'mva_at_position': self._get_mva_at_position(aircraft.x, aircraft.y),
            'current_step': self.current_step,
        }

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return

        try:
            import pygame
        except ImportError:
            raise ImportError(
                "pygame is required for rendering. "
                "Install with: pip install pygame"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("ATC Environment")
            else:
                self.screen = pygame.Surface(
                    (self.screen_width, self.screen_height)
                )
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

        # Clear screen
        self.screen.fill((20, 20, 40))  # Dark blue background

        # Draw grid
        self._draw_grid(pygame)

        # Draw MVAs
        self._draw_mvas(pygame)

        # Draw runway/FAF
        self._draw_runway(pygame)

        # Draw aircraft
        for aircraft in self.aircraft:
            self._draw_aircraft(pygame, aircraft)

        # Draw info panel
        self._draw_info_panel(pygame)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )

    def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        scale = self.screen_width / 100  # 100 NM total width
        screen_x = int((x + 50) * scale)
        screen_y = int((50 - y) * scale)  # Flip y axis
        return screen_x, screen_y

    def _draw_grid(self, pygame):
        """Draw coordinate grid."""
        for i in range(-50, 51, 10):
            # Vertical lines
            x1, y1 = self._world_to_screen(i, -50)
            x2, y2 = self._world_to_screen(i, 50)
            pygame.draw.line(self.screen, (40, 40, 60), (x1, y1), (x2, y2), 1)

            # Horizontal lines
            x1, y1 = self._world_to_screen(-50, i)
            x2, y2 = self._world_to_screen(50, i)
            pygame.draw.line(self.screen, (40, 40, 60), (x1, y1), (x2, y2), 1)

    def _draw_mvas(self, pygame):
        """Draw MVA regions."""
        for mva in self.mvas:
            x1, y1 = self._world_to_screen(mva.x_min, mva.y_max)
            x2, y2 = self._world_to_screen(mva.x_max, mva.y_min)

            # Color based on altitude
            intensity = int(min(255, mva.altitude / 20))
            color = (intensity, 50, 50)

            rect = pygame.Rect(x1, y1, x2 - x1, y2 - y1)
            pygame.draw.rect(self.screen, color, rect, 2)

    def _draw_runway(self, pygame):
        """Draw runway and FAF."""
        # FAF position
        faf_x, faf_y = self._world_to_screen(self.runway.x, self.runway.y)

        # Draw FAF circle
        pygame.draw.circle(self.screen, (0, 255, 0), (faf_x, faf_y), 10, 2)

        # Draw runway direction
        heading_rad = math.radians(self.runway.heading)
        length = 30
        end_x = faf_x - length * math.sin(heading_rad)
        end_y = faf_y - length * math.cos(heading_rad)
        pygame.draw.line(self.screen, (0, 255, 0),
                         (faf_x, faf_y), (int(end_x), int(end_y)), 3)

        # FAF label
        label = self.font.render("FAF", True, (0, 255, 0))
        self.screen.blit(label, (faf_x + 15, faf_y - 10))

    def _draw_aircraft(self, pygame, aircraft: AircraftState):
        """Draw a single aircraft."""
        x, y = self._world_to_screen(aircraft.x, aircraft.y)

        # Aircraft symbol (triangle)
        heading_rad = math.radians(aircraft.heading)
        size = 8

        # Triangle points
        nose = (
            x + size * math.sin(heading_rad),
            y - size * math.cos(heading_rad)
        )
        left = (
            x - size * 0.7 * math.sin(heading_rad - 2.5),
            y + size * 0.7 * math.cos(heading_rad - 2.5)
        )
        right = (
            x - size * 0.7 * math.sin(heading_rad + 2.5),
            y + size * 0.7 * math.cos(heading_rad + 2.5)
        )

        pygame.draw.polygon(self.screen, (255, 255, 0),
                            [nose, left, right])

        # Speed vector
        vector_length = 20
        vec_end = (
            x + vector_length * math.sin(heading_rad),
            y - vector_length * math.cos(heading_rad)
        )
        pygame.draw.line(self.screen, (255, 255, 0), (x, y), vec_end, 1)

        # Data block
        alt_str = f"{int(aircraft.altitude / 100):03d}"  # Flight level format
        hdg_str = f"{int(aircraft.heading):03d}"
        spd_str = f"{int(aircraft.speed)}"

        label = self.font.render(
            f"{aircraft.callsign}", True, (255, 255, 255)
        )
        self.screen.blit(label, (x + 15, y - 20))

        label = self.font.render(
            f"{alt_str} {hdg_str}° {spd_str}kt", True, (200, 200, 200)
        )
        self.screen.blit(label, (x + 15, y))

    def _draw_info_panel(self, pygame):
        """Draw information panel."""
        aircraft = self.aircraft[0]
        info_text = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Distance to FAF: {self._distance_to_faf(aircraft):.1f} NM",
            f"Altitude: {int(aircraft.altitude)} ft",
            f"Heading: {int(aircraft.heading)}°",
            f"MVA: {int(self._get_mva_at_position(aircraft.x, aircraft.y))} ft",
        ]

        y_offset = 10
        for text in info_text:
            label = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(label, (10, y_offset))
            y_offset += 20

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None


# Register the environment
def register_atc_env():
    """Register the ATC environment with Gymnasium."""
    gym.register(
        id="ATCGym-v0",
        entry_point="atc_gym_env:ATCGymEnv",
        max_episode_steps=500,
    )


if __name__ == "__main__":
    # Quick test
    env = ATCGymEnv(render_mode="human")
    obs, info = env.reset()

    print("Observation shape:", obs.shape)
    print("Action space:", env.action_space)
    print("Initial info:", info)

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            print("Episode ended. Final info:", info)
            break

    env.close()