#version notes: this version considers 2km spatial decay and 24h temporal decay.
#area size is 5km * 5km. cell size is 0.2km * 0.2km
#reward function uses a hierarchical bonus strategy to push the UAV towards higher risk cells
#prints the reward/time step history of 5 highest visited cells.
#restricts the starting cells to outer cells (mission can start from any cell on peripheral).
#uses early stopping in evaluation loop to report 5 highest risk cells



import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium import spaces

area_km2 = 25
side_m = int(np.sqrt(area_km2)*1000)
num_sensors = 3000
cell_size_m = 200
grid_cells = side_m // cell_size_m


#sensors random position
np.random.seed(0)
cell_centers_x = (np.arange(grid_cells) + np.random.rand(grid_cells)) * cell_size_m
cell_centers_y = (np.arange(grid_cells) + np.random.rand(grid_cells)) * cell_size_m
grid_x, grid_y = np.meshgrid(cell_centers_x, cell_centers_y)
mandatory_sensors_x = grid_x.flatten()
mandatory_sensors_y = grid_y.flatten()

remaining_sensors = num_sensors - len(mandatory_sensors_x)
extra_sensors_x = np.random.uniform(0, side_m, remaining_sensors)
extra_sensors_y = np.random.uniform(0, side_m, remaining_sensors)

sensor_x = np.concatenate([mandatory_sensors_x, extra_sensors_x])
sensor_y = np.concatenate([mandatory_sensors_y, extra_sensors_y])


cell_x_idx = (sensor_x // cell_size_m).astype(int)
cell_y_idx = (sensor_y // cell_size_m).astype(int)

#Count Sensors in Each Cell
sensor_grid = np.zeros((grid_cells, grid_cells), dtype=int)
for x, y in zip(cell_x_idx, cell_y_idx):
    sensor_grid[y, x] += 1

#Visualization
fig, ax = plt.subplots(figsize=(10, 10))

#Draw grid
for i in range(grid_cells + 1):
    ax.axhline(i * cell_size_m, color='lightgray', linewidth=0.5)
    ax.axvline(i * cell_size_m, color='lightgray', linewidth=0.5)

#Plot sensors
ax.scatter(sensor_x, sensor_y, s=5, color='blue', alpha=0.6, label='Sensors')

#Formatting
ax.set_xlim(0, side_m)
ax.set_ylim(0, side_m)
ax.set_title("Sensor Distribution in 5km x 5km Grid (200m x 200m cells)")
ax.set_xlabel("Meters (X)")
ax.set_ylabel("Meters (Y)")
ax.set_aspect('equal')
plt.legend()
plt.tight_layout()
plt.show()

#Sensors synthetic data
def sensor_value_generator(param, time_min, cell_x, cell_y,
                           fire_cell=(12, 12), fire_time_min=4680,
                           cell_size=200):

    #Base levels
    base_temp = 25  # °C
    base_hum = 60   # %
    base_wind = 5   # m/s
    
    #Max possible effect due to fire build-up
    max_temp_rise = 15   # °C
    max_hum_drop = 40    # %
    max_wind_rise = 20   # m/s

    #Spatial decay
    dx = cell_x - fire_cell[0]
    dy = cell_y - fire_cell[1]
    distance = np.sqrt(dx**2 + dy**2) * cell_size
    spatial_decay = np.exp(-distance / 2000)  # decay over ~2km


    #Temporal decay
    time_to_fire = fire_time_min - time_min
    if time_to_fire < 0:
        time_decay = 0
    else:
        time_decay = np.exp(-time_to_fire / 1440)  # ~24 hours scale

    #Fire factor
    fire_factor = spatial_decay * time_decay

    #Generate value with base + fire effect + small noise
    if param == "temperature":
        value = base_temp + fire_factor * max_temp_rise + np.random.normal(0, 0.5)
    elif param == "humidity":
        value = base_hum - fire_factor * max_hum_drop + np.random.normal(0, 1)
    elif param == "wind_speed":
        value = base_wind + fire_factor * max_wind_rise + np.random.normal(0, 0.5)
    else:
        raise ValueError("Unsupported parameter: choose temperature, humidity, or wind_speed")

    return max(0, value)  #Ensure non-negative


#Fire Environment for UAV
class UAVFireEnv(gym.Env):
    def __init__(self,
                 grid_size=25,
                 fire_cell=(12, 12),
                 fire_time_min=4680,
                 max_steps=2160,
                 time_step_min=2,
                 cell_size=200,
                 w1=0.35, w2=0.35, w3=0.3):
        super(UAVFireEnv, self).__init__()

        self.grid_size = grid_size
        self.fire_cell = fire_cell
        self.fire_time_min = fire_time_min
        self.max_steps = max_steps
        self.time_step_min = time_step_min
        self.cell_size = cell_size
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.enable_early_stop = False
        self.action_space = spaces.Discrete(8)  # 8 movement directions
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_time_min = 0
        edge_indices = [0, 1, 2, 22, 23, 24]
        self.position = (
            self.np_random.choice(edge_indices),
            self.np_random.choice(edge_indices)
        )
        print(f"initial position = {self.position}")

        self.cell_log = {}             # history of cell visits
        self.best_cell = None          # store best predicted cell
        self.best_score = 0            # store best reward+bonus seen
        self.best_time = None          # time it was found

        return self._get_observation(), {}


    def step(self, action):
        self.current_step += 1
        self.current_time_min = self.current_step * self.time_step_min
        self._move(action)
        obs = self._get_observation()
        reward = self._calculate_reward(obs)

        #Early prediction trigger: same high-score cell visited 3+ times
        terminated = False
        cell = self.position

        if self.enable_early_stop:
            confident_cells = []

            for cell, history in self.cell_log.items():
                high_reward_visits = [r for _, r in history if r >= 1.0]
                if len(high_reward_visits) >= 3:
                    total_reward = sum(r for _, r in history)  #sum full reward history
                    confident_cells.append((cell, total_reward))

            if len(confident_cells) == 5:
                #Sort by total accumulated reward
                confident_cells.sort(key=lambda x: x[1], reverse=True)

                terminated = True
                time_to_fire = self.fire_time_min - self.current_time_min

                print("\n UAV identified 5 high-confidence cells:")
                for i, (cell, total_reward) in enumerate(confident_cells, start=1):
                    print(f"  {i}. Cell {cell}, Total Reward (all visits): {total_reward:.2f}")

                print(f"\n Search stopped at time: {self.current_time_min} min")
                print(f"Time remaining to fire: {time_to_fire} min ({time_to_fire / 60:.1f} hrs)\n")
        
        truncated = self.current_step >= self.max_steps
        info = {}

        return obs, reward, terminated, truncated, info


    def _move(self, action):
        x, y = self.position
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),  # N, S, E, W
            (1, 1), (-1, 1), (1, -1), (-1, -1)  # NE, NW, SE, SW
        ]
        dx, dy = directions[action]
        new_x, new_y = x + dx, y + dy

        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            self.position = (new_x, new_y)

    def _get_observation(self):
        x, y = self.position
        temp = sensor_value_generator("temperature", self.current_time_min, x, y)
        hum = sensor_value_generator("humidity", self.current_time_min, x, y)
        wind = sensor_value_generator("wind_speed", self.current_time_min, x, y)

        obs = np.array([
            x / self.grid_size,
            y / self.grid_size,
            temp / 40.0,
            hum / 100.0,
            wind / 20.0
        ], dtype=np.float32)
        return obs

    def _calculate_reward(self, obs):
        x, y = self.position
        cell = (x, y)
        temp = obs[2] * 40
        hum = obs[3] * 100
        wind = obs[4] * 20
        score = self.w1 * temp / 40 + self.w2 * (1 - hum / 100) + self.w3 * wind / 20
        frs = 1 / (1 + np.exp(-score))

        bonus = 0
        if frs >= 0.65 and frs < 0.75:
            bonus = 0.4
        if frs >= 0.75 and frs < 0.85:
            bonus = 0.8
        if frs >= 0.85:
            bonus = 1.6
  
        prior_visits = self.cell_log.get(cell, [])
        penalty = -0.1 if len(prior_visits) >= 2 and frs < 0.61 else 0
        total_reward = frs + bonus + penalty
        self.cell_log.setdefault(cell, []).append((self.current_step, total_reward))

    #Track best cell for early prediction logic
        if total_reward > self.best_score:
            self.best_score = total_reward
            self.best_cell = cell
            self.best_time = self.current_time_min

        return total_reward

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Position: {self.position}, Time: {self.current_time_min} min")


from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

#Check if environment is valid
check_env(UAVFireEnv(), warn=True)

#Instantiate environment
env = UAVFireEnv()

#Instantiate PPO model with minimal network for low-resource devices
policy_kwargs = dict(net_arch=[32, 32])  # smaller network for microcontroller compatibility
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

#Train the model (can reduce timesteps for quick testing)
model.learn(total_timesteps=50000)

#Save model
model.save("ppo_uav_fire_predictor")

#Reset env for post-training demo
env.enable_early_stop = True
obs, _ = env.reset(seed=0)

positions = [env.position]
rewards = []
temps = []
hums = []
winds = []
times = []

for _ in range(env.max_steps):
    #print(f"Time Step: {env.current_step}, Position: {env.position}")
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    rewards.append(reward)
    temps.append(obs[2] * 40)
    hums.append(obs[3] * 100)
    winds.append(obs[4] * 20)
    positions.append(env.position)
    times.append(env.current_step)
    #print(f"Temp: {temps[-1]:.2f} °C, Humidity: {hums[-1]:.2f} %, Wind: {winds[-1]:.2f} m/s, Reward: {rewards[-1]:.4f}")
    if done:
        break

grid_size = env.grid_size
grid = np.zeros((grid_size, grid_size))
for x, y in positions:
    grid[y, x] += 1

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(grid, cmap='Blues', origin='lower')
ax.set_title("UAV Path on Grid")
ax.set_xlabel("X Cell")
ax.set_ylabel("Y Cell")
ax.plot([p[0] for p in positions], [p[1] for p in positions], color='red', marker='o', linestyle='-')
plt.grid(True)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(times, rewards, 'r-', label='Reward (FRS)')
ax.set_ylabel("Reward", color='r')
ax.set_xlabel("Time Step")
ax.tick_params(axis='y', labelcolor='r')

available = len(rewards)
num_to_annotate = min(4, available // 2)
if num_to_annotate > 0:
    sorted_indices = np.argsort(rewards)
    highlight_indices = sorted_indices[:num_to_annotate].tolist() + sorted_indices[-num_to_annotate:].tolist()
    for i in highlight_indices:
        if i < len(times):
            ax.annotate(f"({temps[i]:.1f},{hums[i]:.1f},{winds[i]:.1f})",
                        (times[i], rewards[i]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center', fontsize=8, color='black')

plt.title("Reward over Time with Sensor Annotations")
plt.legend()
plt.tight_layout()
plt.show()
