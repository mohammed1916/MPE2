# noqa: D212, D415
"""
# Simple Tag

```{figure} mpe2/mpe2_simple_tag.gif
:width: 140px
:name: simple_tag
```

This environment is part of the <a href='https://mpe2.farama.org/mpe2/'>MPE environments</a>. Please read that page first for general information.

| Import             |              `from mpe2 import simple_tag_no_collision_detect_radius`              |
|--------------------|------------------------------------------------------------|
| Actions            | Discrete/Continuous                                        |
| Parallel API       | Yes                                                        |
| Manual Control     | No                                                         |
| Agents             | `agents= [adversary_0, adversary_1, adversary_2, agent_0]` |
| Agents             | 4                                                          |
| Action Shape       | (5)                                                        |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (50))                            |
| Observation Shape  | (14),(16)                                                  |
| Observation Values | (-inf,inf)                                                 |
| State Shape        | (62,)                                                      |
| State Values       | (-inf,inf)                                                 |


This is a predator-prey environment. Good agents (green) are faster and receive a negative reward for being hit by adversaries (red) (-10 for each collision). Adversaries are slower and are rewarded for hitting good agents (+10 for each collision). Obstacles (large black circles) block the way. By
default, there is 1 good agent, 3 adversaries and 2 obstacles.

So that good agents don't run to infinity, they are also penalized for exiting the area by the following function:

``` python
def bound(x):
      if x < 0.9:
          return 0
      if x < 1.0:
          return (x - 0.9) * 10
      return min(np.exp(2 * x - 2), 10)
```

Agent and adversary observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]`

Agent and adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_tag_no_collision_detect_radius.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False, dynamic_rescaling=False)
```



`num_good`:  number of good agents

`num_adversaries`:  number of adversaries

`num_obstacles`:  number of obstacles

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

`dynamic_rescaling`: Whether to rescale the size of agents and landmarks based on the screen size

"""

import numpy as np
import logging
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn

from mpe2._mpe_utils.core import Agent, Landmark, World
from mpe2._mpe_utils.scenario import BaseScenario
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env

# try:
import shapely.geometry as sg
# except ImportError:
    # sg = None


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_good=3,
        num_adversaries=2,
        num_obstacles=0,
        continuous_actions=True,
        max_cycles=20,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            max_cycles= max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_good, num_adversaries, num_obstacles)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "simple_tag_no_collision_detect_radius"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_good=3, num_adversaries=2, num_obstacles=0):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            # agent.size = 0.075 if agent.adversary else 0.05
            agent.size = 0.075*2 if agent.adversary else 0.05*2
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.2*2
            landmark.boundary = False
        # add polygons for negative space boundaries
        # if sg:
        world.polygons = [sg.Polygon([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])]
        # else:
            # world.polygons = []
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.85, 0.35])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.5, 1, 1])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        logger = logging.getLogger(__name__)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                # Only randomize landmark positions on the first reset. Subsequent
                # calls to `reset_world` will keep the same landmark positions
                # (but reset landmark velocity). This prevents landmarks from
                # moving every episode while preserving initial randomness.
                if not getattr(landmark, "_init_pos_set", False):
                    landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                    landmark._init_pos_set = True
                    print("Changing Landmark position: ", landmark.state.p_pos)
                    # use debug-level logging so position changes do not flood stdout by default
                    # logger.debug("Changing Landmark position: %s", landmark.state.p_pos)
                # Always reset velocity to zero on each world reset
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        # Do not collide with same team
        # if agent1.adversary == agent2.adversary:
        #     return False

        # Compute distance between agents
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size

        # If within collision range
        if dist < dist_min:
            # Stop the good agent’s motion if caught
            # if not agent1.adversary:
            #     agent1.state.p_vel *= 0
            # elif not agent2.adversary:
            #     agent2.state.p_vel *= 0
            return True

        return False

    
    def same_team_penalty(self, agent, world, radius=0.2):
        penalty = 0
        for other in world.agents:
            if other is agent or other.adversary != agent.adversary:
                continue
            dist = np.linalg.norm(agent.state.p_pos - other.state.p_pos)
            if dist < radius:
                penalty -= (radius - dist)  # closer -> bigger penalty
        return penalty


    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        main_reward += self.same_team_penalty(agent, world, radius=0.2)
        return main_reward

    def agent_reward(self, agent, world):
        rew = 0
        adversaries = self.adversaries(world)

        # Penalize if caught by adversaries
        for a in adversaries:
            if self.is_collision(a, agent):
                rew -= 10
        # Penalize if distance is small to adversaries
        for a in adversaries:
            dist = np.linalg.norm(agent.state.p_pos - a.state.p_pos)
            rew -= 0.1 * (1 - dist)

        # Reward for being near the landmark
        dists = [np.linalg.norm(agent.state.p_pos - l.state.p_pos) for l in world.landmarks]
        min_dist = min(dists)
        rew += 5 * (1 - min_dist)
        

        # Penalize for proximity to boundary (treating as obstacle)
        # if sg and world.polygons:
        
        for poly in world.polygons:
            dist = poly.boundary.distance(sg.Point(agent.state.p_pos))
            if dist < 0.2:
                rew -= 20 * (0.2 - dist) / 0.2  # linear penalty for being too close to boundary
        
        # else:
        #     # Fallback to square bounds
        #     def bound(x):
        #         if x < 0.7:
        #             return 0
        #         if x < 0.9:
        #             return (x - 0.7) * 5  # penalty starts earlier
        #         if x < 1.0:
        #             return (x - 0.9) * 20  # sharper penalty near edge
        #         return min(np.exp(3 * x - 2), 20)  # steeper exponential

        #     for p in range(world.dim_p):
        #         x = abs(agent.state.p_pos[p])
        #         rew -= bound(x)

        return rew


    def adversary_reward(self, agent, world):
        rew = 0
        shape = True
        agents = self.good_agents(world)

        for ag in agents:
            dist = np.linalg.norm(agent.state.p_pos - ag.state.p_pos)
            if shape:
                rew += 0.5 * (1 - dist)
            if agent.collide and self.is_collision(agent, ag):
                rew += 10
        return rew

    def observation(self, agent, world):
        # Radius-based observation for fixed size
        radius = 1.0  # observation radius
        max_agents = 3  # max other agents to observe
        max_landmarks = 1  # max landmarks to observe
        
        # Self state
        obs = [agent.state.p_vel, agent.state.p_pos]
        
        # Add direction to landmark (relative position vector)
        landmark = world.landmarks[0]  # assuming single landmark
        rel_pos_to_landmark = landmark.state.p_pos - agent.state.p_pos
        obs.append(rel_pos_to_landmark)
        
        # Add angle to landmark
        angle_to_landmark = np.arctan2(rel_pos_to_landmark[1], rel_pos_to_landmark[0])
        obs.append(np.array([angle_to_landmark]))
        
        # Add distance to boundary (treating boundary as obstacle)
        dist_to_boundary = world.polygons[0].boundary.distance(sg.Point(agent.state.p_pos))
        obs.append(np.array([dist_to_boundary]))
        
        # Get nearby landmarks
        landmark_infos = []
        for lm in world.landmarks:
            if not lm.boundary:
                rel_pos = lm.state.p_pos - agent.state.p_pos
                dist = np.linalg.norm(rel_pos)
                if dist <= radius:
                    landmark_infos.append((dist, rel_pos, 0))  # type=0 for landmark
        
        # Sort by distance and take closest
        landmark_infos.sort(key=lambda x: x[0])
        for i in range(max_landmarks):
            if i < len(landmark_infos):
                obs.extend([landmark_infos[i][1], np.array([landmark_infos[i][2]])])  # pos, type
            else:
                obs.extend([np.zeros(2), np.array([0])])  # pad
        
        # Get nearby other agents
        agent_infos = []
        for other in world.agents:
            if other is agent:
                continue
            rel_pos = other.state.p_pos - agent.state.p_pos
            dist = np.linalg.norm(rel_pos)
            if dist <= radius:
                vel = other.state.p_vel if not other.adversary else np.zeros(2)
                type_val = 1 if not other.adversary else 2  # 1=good, 2=adversary
                agent_infos.append((dist, rel_pos, vel, type_val))
        
        # Sort by distance and take closest
        agent_infos.sort(key=lambda x: x[0])
        for i in range(max_agents):
            if i < len(agent_infos):
                obs.extend([agent_infos[i][1], agent_infos[i][2], np.array([agent_infos[i][3]])])  # pos, vel, type
            else:
                obs.extend([np.zeros(2), np.zeros(2), np.array([0])])  # pad
        
        return np.concatenate(obs)

    def is_goal_reached(self, world, required_count=1, eps=1e-6):
        """Return True if at least `required_count` unique good agents are within
        contact distance of any non-boundary landmark.
        """
        reached = set()

        for ag in world.agents:
            # only consider good agents (not adversaries)
            if ag.adversary:
                continue

            ag_id = ag.name

            for lm in world.landmarks:
                if lm.boundary:
                    continue
                # Euclidean distance
                dist = np.linalg.norm(ag.state.p_pos - lm.state.p_pos)
                # If within contact distance (agent.size + landmark.size)
                if dist <= (ag.size + lm.size + eps):
                    reached.add(ag_id)
                    # short-circuit if we've reached the required count
                    if len(reached) >= required_count:
                        return True
        return False
