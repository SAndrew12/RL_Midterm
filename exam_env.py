#%%

# how to construct an open ai gym environment from scratch: https://www.gymlibrary.dev/content/environment_creation/

import gymnasium as gym
import pygame
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    
    metadata = {"render_modes": ["human"], "render_fps": 10} # "human" render_mode means env. is designed to render in a format that a human can see real time.
    
    def __init__(self, render_mode=None, size=3, seed = 123, max_barriers=20):
        
        # Pygame Variables
        # ----------------------------------------------------------------
        pygame.init()
        pygame.font.init()
        self.size = size  
        self.window_size = 512
        self.render_mode = render_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window = None  
        self.clock = None
        self.font = pygame.font.Font(None, 36) 
        
        # Open AI Variables
        # ----------------------------------------------------------------
        self.observation_space = spaces.Discrete(self.size * self.size)
        self.action_space = spaces.Discrete(4)
        self.action_to_direction = {
            0: np.array([1,0]),  # down
            1: np.array([0,1]),  # right
            2: np.array([-1,0]), # up
            3: np.array([0,-1]), # left
        }
        
        # Metrics
        # ----------------------------------------------------------------
        self.q = None
        self.episode = 0
        self.steps = 0
        self.steps_per_episode = []
        self.average_steps = 0
        self.average_steps_per_episode = []
        self.cum_reward = 0
        self.rewards = []
        
        # Exam 1: Barriers 
        # ----------------------------------------------------------------
        self.max_barriers = max_barriers  
        self.barrier_increment = 1  
        
    def reset(self, seed=None):
        super().reset(seed=seed) 
        
        # Initializing self.q
        # ----------------------------------------------------------------
        if self.q is None:
            self.q = np.zeros((self.observation_space.n, self.action_space.n))
                
        # Return and reset state and terminal state
        # ----------------------------------------------------------------
        self.state = np.array([0,0])
        self.terminal_state = np.array([self.size - 1, self.size - 1])
        observation = self.convert_2d_to_1d(self.state)
        
        # Exam 1: Barriers 
        # ----------------------------------------------------------------
        self._update_barriers()
        
        # Reset steps counter and add episode
        # ----------------------------------------------------------------
        self.cum_reward = 0
        self.steps = 0
        self.episode += 1
        
        # Return Pygame Frame
        # ----------------------------------------------------------------
        if self.render_mode == "human":
            self.render()
            
        return observation
    
    def _update_barriers(self):

        self.barrier_locations = []
        
        num_barriers = min(self.episode * self.barrier_increment, self.max_barriers)

        grid = np.zeros((self.size, self.size), dtype=int)
        
        path = [[row, col] for row in range(self.size) for col in range(1,self.size) if row == col or row == col-1]

        grid[0, 0] = 0 
        grid[self.size - 1, self.size - 1] = 0  

        while len(self.barrier_locations) < num_barriers:
            random_row = np.random.randint(low=0, high=self.size)  
            random_col = np.random.randint(low=0, high=self.size)  

            if ([random_row, random_col] not in self.barrier_locations and 
                not np.array_equal([random_row, random_col], self.state) and 
                not np.array_equal([random_row, random_col], self.terminal_state)):
                
                grid[random_row, random_col] = 1  
                self.barrier_locations.append([random_row, random_col])

                if [random_row, random_col] in path:
                    grid[random_row, random_col] = 0
                    self.barrier_locations.remove([random_row, random_col])
    
    def step(self, action):
        
        self.steps += 1

        # Return Environment Reward Calculation  
        # ----------------------------------------------------------------
        if type(self.terminal_state) == int:
            self.terminal_state = self.convert_1d_to_2d(self.terminal_state)
        if type(self.state) == int:
            self.state = self.convert_1d_to_2d(self.state)
        
        directions = [
            ([1, 0], self.state[0] + 1 < self.size),   # down
            ([0, 1], self.state[1] + 1 < self.size),   # right
            ([-1, 0], self.state[0] - 1 >= 0),         # up
            ([0, -1], self.state[1] - 1 >= 0)          # left
        ]
        
        values = [
            10 if np.array_equal([self.state[0] + d[0], self.state[1] + d[1]], self.terminal_state) 
            else -1 if any(np.array_equal([self.state[0] + d[0], self.state[1] + d[1]], b) for b in self.barrier_locations)
            else 0 if valid 
            else -1 
            for d, valid in directions
        ]
        reward = values[action]
        self.cum_reward += reward
        step = directions[action][0]
        new_location = ([self.state[0] + step[0], self.state[1] + step[1]] if values[action] != -1 else self.state)
        if not any(np.array_equal(new_location, barrier) for barrier in self.barrier_locations):
            self.state = new_location
        terminated = np.array_equal(self.state, self.terminal_state)
        observation = self.convert_2d_to_1d(self.state)
    
        # Metrics
        # ----------------------------------------------------------------
        if terminated:
            self.rewards.append(self.cum_reward)
            self.steps_per_episode.append(self.steps)
            self.average = np.mean(self.steps_per_episode)
            self.average_steps_per_episode.append(self.average)
            print(f"Average Steps: {self.average.__round__(2)} | Average Reward: {np.mean(self.rewards).__round__(2)}") 
        
        # Render Pygame Frame
        # ----------------------------------------------------------------
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated

    def render(self):
        
        # Initialize self.q
        # ----------------------------------------------------------------
        q = np.array(self.q).T.reshape(self.action_space.n, self.size, self.size)
        if type(self.terminal_state) == int:
            self.terminal_state = self.convert_1d_to_2d(self.terminal_state)
        if type(self.state) == int:
            self.state = self.convert_1d_to_2d(self.state)
        
        # Initializing Pygame Window (Frame) and Clock (FPS)
        # ----------------------------------------------------------------
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size + 70))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Pygame Frame 
        # ----------------------------------------------------------------    
        frame = pygame.Surface((self.window_size, self.window_size + 70))
        frame.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size 

        # Draw Terminal State
        # ---------------------------------------------------------------- 
        pygame.draw.rect(
            frame,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.terminal_state,
                (pix_square_size, pix_square_size),
            ),
        )
        
        # Draw Agent
        # ---------------------------------------------------------------- 
        pygame.draw.circle(
            frame,
            (0, 0, 0), 
            (int(self.state[1] * pix_square_size + 0.5 * pix_square_size), 
            int(self.state[0] * pix_square_size + 0.5 * pix_square_size)),
            int(pix_square_size / 3) + 3, 
        )
        pygame.draw.circle(
            frame,
            (0, 0, 255),  
            (int(self.state[1] * pix_square_size + 0.5 * pix_square_size), 
            int(self.state[0] * pix_square_size + 0.5 * pix_square_size)),
            pix_square_size / 3,
        )
        
        # Draw Barriers
        # ---------------------------------------------------------------- 
        for barrier in self.barrier_locations:
            barrier_x = round(barrier[1] * pix_square_size)
            barrier_y = round(barrier[0] * pix_square_size)
            
            pygame.draw.rect(
                frame,
                (0, 0, 0),
                pygame.Rect(
                    (barrier_x, barrier_y),
                    (round(pix_square_size), round(pix_square_size)),  # Ensure it fully covers the box
                ),
            )

        # Draw Policy 
        # ---------------------------------------------------------------- 
        arrow_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        arrow_surface.set_alpha(128) 
        for row in range(self.size):
            for col in range(self.size):
                if np.array_equal([row, col], self.terminal_state):
                    continue 

                center_x = int(col * pix_square_size + 0.5 * pix_square_size)
                center_y = int(row * pix_square_size + 0.5 * pix_square_size)
                
                if self.episode == 1:
                    values = [0 for d in self.action_to_direction.values()]
                else: 
                    values = q[:,row, col] 
                max_val = np.max(values)
                directions = [i for i, val in enumerate(values) if val == max_val]

                for direction in directions:
                    arrow_length = int(pix_square_size / 4)  # Adjust the arrow length based on grid square size
                    arrow_width = int(pix_square_size / 12)  # Adjust the arrow width

                    if direction == 0:  # Down
                        start_pos = (center_x, center_y - arrow_length)
                        end_pos = (center_x, center_y + arrow_length)
                        arrow_tip = [(center_x - arrow_width, center_y + arrow_length),
                                    (center_x + arrow_width, center_y + arrow_length),
                                    (center_x, center_y + arrow_length + arrow_width)]
                    elif direction == 1:  # Right
                        start_pos = (center_x - arrow_length, center_y)
                        end_pos = (center_x + arrow_length, center_y)
                        arrow_tip = [(center_x + arrow_length, center_y - arrow_width),
                                    (center_x + arrow_length, center_y + arrow_width),
                                    (center_x + arrow_length + arrow_width, center_y)]
                    elif direction == 2:  # Up
                        start_pos = (center_x, center_y + arrow_length)
                        end_pos = (center_x, center_y - arrow_length)
                        arrow_tip = [(center_x - arrow_width, center_y - arrow_length),
                                    (center_x + arrow_width, center_y - arrow_length),
                                    (center_x, center_y - arrow_length - arrow_width)]
                    elif direction == 3:  # Left
                        start_pos = (center_x + arrow_length, center_y)
                        end_pos = (center_x - arrow_length, center_y)
                        arrow_tip = [(center_x - arrow_length, center_y - arrow_width),
                                    (center_x - arrow_length, center_y + arrow_width),
                                    (center_x - arrow_length - arrow_width, center_y)]
                    
                    if not (0 <= start_pos[0] < self.window_size and 0 <= start_pos[1] < self.window_size):
                        continue
                    if not (0 <= end_pos[0] < self.window_size and 0 <= end_pos[1] < self.window_size):
                        end_pos = (min(max(end_pos[0], 0), self.window_size), min(max(end_pos[1], 0), self.window_size))

                    pygame.draw.line(arrow_surface, (0, 0, 0), start_pos, end_pos, 3)
                    pygame.draw.polygon(arrow_surface, (0, 0, 0), arrow_tip)
                    
                        
        frame.blit(arrow_surface, (0, 0))
        
        # Draw Gridlines 
        # ---------------------------------------------------------------- 
        for x in range(self.size + 1):
            pygame.draw.line(
                frame,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                frame,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
            )
        
        # Draw Metrics
        # ---------------------------------------------------------------- 
        episode_text = self.font.render(f'Episode: {self.episode}', True, 0)
        step_text = self.font.render(f'Steps: {self.steps}', True, 0)
        title = self.font.render(f'Exam 1', True, 0)

        text_x = 10
        text_y = self.window_size + 10
        step_text_y = self.window_size + 40

        frame.blit(episode_text, (text_x, text_y))
        frame.blit(step_text, (text_x, step_text_y))
        frame.blit(title, (text_x + 200, step_text_y - 20))
                        
        if self.render_mode == "human":
            self.window.blit(frame, frame.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def convert_1d_to_2d(self, indx):
        return np.array([indx // self.size, indx % self.size])
    
    def convert_2d_to_1d(self, array):
        return (array[0] * self.size) + array[1]