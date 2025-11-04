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
        max_cycles=25,
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
            # agent.collide = True
            agent.collide = False
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05
            # agent.size = 0.075*2 if agent.adversary else 0.05*2
            agent.accel = 3.0 if agent.adversary else 4.0
            # agent.accel = 4.0 if agent.adversary else 0.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
            # agent.max_speed = 1.0 if agent.adversary else 0.0
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
                    landmark.state.p_pos = np_random.uniform(-0.9, 0.9, world.dim_p)
                    # landmark.state.p_pos = np_random.uniform(0.2, 0.7, world.dim_p)
                    # landmark.state.p_pos = np.array([-0.532, 0.387]) if np.random.rand() < 0.5 else np.array([-0.932, -0.387])
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
                print("Skipping agent in same_team_penalty")
                continue
            print("Calculating distance for same_team_penalty")
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
        # main_reward += self.same_team_penalty(agent, world, radius=0.2)
        return main_reward

    def agent_reward(self, agent, world):
        rew = 0
        adversaries = self.adversaries(world)

        """
        # Penalize if caught by adversaries
        for a in adversaries:
            if self.is_collision(a, agent):
                rew -= 10
        # Penalize if distance is small to adversaries
        for a in adversaries:
            dist = np.linalg.norm(agent.state.p_pos - a.state.p_pos)
            rew -= 0.1 * (2 - dist)
        """

        # Reward for being near the landmark
        # it is not agent.adversary as called in agent_reward
        dists = [np.linalg.norm(agent.state.p_pos - l.state.p_pos) for l in world.landmarks]
        min_dist = min(dists)
        rew += 5 * (2 - min_dist)
    

        # Penalize for proximity to boundary (treating as obstacle)
        if sg and world.polygons:
            for poly in world.polygons:
                dist = poly.boundary.distance(sg.Point(agent.state.p_pos))
                if dist < 0.15:
                    rew -= 10 * (0.15 - dist) / 0.15  # linear penalty for being too close to boundary
        else:
            # Fallback to square bounds
            def bound(x):
                if x < 0.7:
                    return 0
                if x < 0.9:
                    return (x - 0.7) * 5  # penalty starts earlier
                if x < 1.0:
                    return (x - 0.9) * 20  # sharper penalty near edge
                return min(np.exp(3 * x - 2), 20)  # steeper exponential

            for p in range(world.dim_p):
                x = abs(agent.state.p_pos[p])
                rew -= bound(x)

        return rew


    def adversary_reward(self, agent, world):
        rew = 0
        shape = True
        agents = self.good_agents(world)

        for ag in agents:
            # print(f"Positions - {agent.name}: {agent.state.p_pos}, {ag.name}: {ag.state.p_pos}")
            dist = np.linalg.norm(agent.state.p_pos - ag.state.p_pos)
            if shape:
                rew += 5 * (2 - dist)
                # print(f"Distance between {agent.name} and {ag.name}: ", dist)
                # print("Adversary shape reward: ",rew,": ", 5 * (2 - dist))
            # if agent.collide and self.is_collision(agent, ag):
            if self.is_collision(agent, ag):
                if not ag.adversary:
                    ag.color = np.array([0, 0, 0])
                    print(f"{agent.name} caught {ag.name}!")
                    rew += 100
        
        # Penalize adversaries for going near boundaries too
        if sg and world.polygons:
            for poly in world.polygons:
                dist = poly.boundary.distance(sg.Point(agent.state.p_pos))
                if dist < 0.15:
                    rew -= 10 * (0.15 - dist) / 0.15
        else:
            def bound(x):
                if x < 0.7:
                    return 0
                if x < 0.9:
                    return (x - 0.7) * 5
                if x < 1.0:
                    return (x - 0.9) * 20
                return min(np.exp(3 * x - 2), 20)

            for p in range(world.dim_p):
                x = abs(agent.state.p_pos[p])
                rew -= bound(x)
        
        """
        for adversary agents
        # distance for adversaries to good agents, more rewards for adversaries if closer they are, so that they can catch them
        good_agents = self.good_agents(world)
        for a in adversaries:
            for ag in good_agents:
                dist = np.linalg.norm(a.state.p_pos - ag.state.p_pos)
                rew += 5 * (2 - dist)
        """
        return rew

    def observation(self, agent, world):
        # Radius-based observation for fixed size
        radius = 10.0  # observation radius
        max_agents = 3  # max other agents to observe
        max_landmarks = 1  # max landmarks to observe
        
        # Self state
        # obs = [agent.state.p_vel, agent.state.p_pos]
        # obs = [agent.state.p_pos]
        obs = []
        # obs_info = {"Self Velocity": agent.state.p_vel.shape, "Self Position": agent.state.p_pos.shape}
        
        # Add boundary distance to observation
        if sg and world.polygons:
            dist_to_boundary = world.polygons[0].boundary.distance(sg.Point(agent.state.p_pos))
            obs.append(np.array([dist_to_boundary]))
        else:
            # Compute minimum distance to any boundary edge
            min_boundary_dist = min(1.0 - abs(agent.state.p_pos[0]), 1.0 - abs(agent.state.p_pos[1]))
            obs.append(np.array([min_boundary_dist]))
        
        landmark = world.landmarks[0]  # assuming single landmark
        rel_pos_to_landmark = landmark.state.p_pos - agent.state.p_pos
        obs.append(rel_pos_to_landmark)
        
        
        good_agents = self.good_agents(world)
        adversary_agents = self.adversaries(world)

        # Assuming relative position with itself will be zero, it will be learning that it's info is in that position
        for ga in good_agents + adversary_agents:
                rel_pos = ga.state.p_pos - agent.state.p_pos
                # dist = np.linalg.norm(rel_pos)
                # if dist <= radius:
                obs.append(rel_pos)
            
 

        """
        # Boundary
        # Add distance to boundary (treating boundary as obstacle)
        dist_to_boundary = world.polygons[0].boundary.distance(sg.Point(agent.state.p_pos))
        obs.append(np.array([dist_to_boundary]))
        """
        """
        # Add direction to landmark (relative position vector)
        if agent.adversary:
            # For adversary, direction to nearest good agent
            # good_agents = self.good_agents(world)
            # if good_agents:
            #     nearest_good_agent = min(good_agents, key=lambda a: np.linalg.norm(a.state.p_pos - agent.state.p_pos))
            #     rel_pos_to_nearest_good = nearest_good_agent.state.p_pos - agent.state.p_pos
            #     obs.append(rel_pos_to_nearest_good)
            
            # Add angle to agent
            # If good agent angle to landmark else angle to nearest adversary
            # Find nearest good agent
            good_agents = self.good_agents(world)
            # if good_agents:
            #     nearest_good_agent = min(good_agents, key=lambda a: np.linalg.norm(a.state.p_pos - agent.state.p_pos))
            #     rel_pos_to_nearest_good = nearest_good_agent.state.p_pos - agent.state.p_pos
            #     obs.append(rel_pos_to_nearest_good)
            #     obs_info["1.Rel Pos to Nearest Good Agent"] = rel_pos_to_nearest_good.shape
                # angle_to_nearest_good = np.arctan2(rel_pos_to_nearest_good[1], rel_pos_to_nearest_good[0])
                # obs.append(np.array([angle_to_nearest_good]))
                # obs_info["Angle to Nearest Good Agent"] = angle_to_nearest_good
            for ga in good_agents:
                rel_pos = ga.state.p_pos - agent.state.p_pos
                dist = np.linalg.norm(rel_pos)
                # if dist <= radius:
                obs.append(rel_pos)
                    # obs_info["1.Rel Pos to Good Agent"] = rel_pos.shape
                    # angle_to_good = np.arctan2(rel_pos[1], rel_pos[0])
                    # obs.append(np.array([angle_to_good]))
                    # obs_info["1.Angle to Good Agent"] = angle_to_good
        else:
            landmark = world.landmarks[0]  # assuming single landmark
            rel_pos_to_landmark = landmark.state.p_pos - agent.state.p_pos
            obs.append(rel_pos_to_landmark)
            # obs_info["1.Rel Pos to Landmark"] = rel_pos_to_landmark.shape
            
            # # Angle to landmark
            # angle_to_landmark = np.arctan2(rel_pos_to_landmark[1], rel_pos_to_landmark[0])
            # obs.append(np.array([angle_to_landmark]))
            # obs_info["1.Angle to Landmark"] = angle_to_landmark
        
        # print("Observation before nearby entities: ", obs)
        # print(f"Agent: {agent.name}, Adversary: {agent.adversary}")
        """
        
        """    
        # Get nearby landmarks
        if not agent.adversary:
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
                    obs_info["2. Rel Pos to Landmark"] = landmark_infos[i][1].shape
                    obs_info["2. Type Landmark"] = np.array([landmark_infos[i][2]]).shape
                else:
                    obs.extend([np.zeros(2), np.array([0])])  # pad
                    obs_info["2. Rel Pos to Landmark Pad"] = np.zeros(2).shape
                    obs_info["2. Type Landmark Pad"] = np.array([0]).shape
        else:
            good_agentInfos = []
            for ga in self.good_agents(world):
                rel_pos = ga.state.p_pos - agent.state.p_pos
                dist = np.linalg.norm(rel_pos)
                if dist <= radius:
                    vel = ga.state.p_vel
                    type_val = 1  # type=1 for good agent
                    # good_agentInfos.append((dist, rel_pos, vel, type_val))
                    good_agentInfos.append((dist, rel_pos, 0))
            
            # Sort by distance and take closest
            good_agentInfos.sort(key=lambda x: x[0])
            for i in range(len(self.good_agents(world))):
                if i < len(good_agentInfos):
                    # obs.extend([good_agentInfos[i][1], good_agentInfos[i][2], np.array([good_agentInfos[i][3]])])  # pos, vel, type
                    obs.extend([good_agentInfos[i][1], np.array([good_agentInfos[i][2]])])  # pos, vel, type
                    obs_info["3. Rel Pos to Good Agent"] = good_agentInfos[i][1].shape
                    obs_info["3. Type Good Agent"] = np.array([good_agentInfos[i][2]]).shape
                else:
                    obs.extend([np.zeros(2), np.zeros(1)])  # pad
                    obs_info["3. Rel Pos to Good Agent Pad"] = np.zeros(2).shape
                    obs_info["3. Type Good Agent Pad"] = np.array([0]).shape
        """

        # print("Observation before nearby entities: ", obs)
        # print(f"Agent: {agent.name}, Adversary: {agent.adversary}")

        # # Get nearby other agents
        # agent_infos = []
        # for other in world.agents:
        #     if other is agent:
        #         continue
        #     rel_pos = other.state.p_pos - agent.state.p_pos
        #     dist = np.linalg.norm(rel_pos)
        #     if dist <= radius:
        #         vel = other.state.p_vel if not other.adversary else np.zeros(2)
        #         type_val = 1 if not other.adversary else 2  # 1=good, 2=adversary
        #         agent_infos.append((dist, rel_pos, vel, type_val))
        
        # # Sort by distance and take closest
        # agent_infos.sort(key=lambda x: x[0])
        # for i in range(max_agents):
        #     if i < len(agent_infos):
        #         obs.extend([agent_infos[i][1], agent_infos[i][2], np.array([agent_infos[i][3]])])  # pos, vel, type
        #     else:
        #         obs.extend([np.zeros(2), np.zeros(2), np.array([0])])  # pad
        obs_concatenated = np.concatenate(obs)
        # print("Observation Info: ", agent.name, obs_concatenated.shape, obs_info, obs_concatenated)
        return obs_concatenated

    def is_goal_reached(self, world, required_count=3, eps=1e-6, vis=False):
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
                if dist <= (ag.size + lm.size):
                    reached.add(ag_id)
                    ag.color = np.array([0.6, 1, 1])
                    # if vis:
                        # ag.max_speed = 0.0  # stop moving once reached
                    # short-circuit if we've reached the required count
                    if len(reached) >= required_count:
                        return True
        
        # Training for Adversary if it collides with good agents
        # for ag in world.agents: 
        #     # only consider adversary agents
        #     if not ag.adversary:
        #         continue

        #     ag_id = ag.name

        #     for good_ag in self.good_agents(world):
        #         # Euclidean distance
        #         dist = np.linalg.norm(ag.state.p_pos - good_ag.state.p_pos)
        #         # If within contact distance (agent.size + landmark.size)
        #         if dist <= (ag.size + good_ag.size):
        #             reached.add(ag_id)
        #             # short-circuit if we've reached the required count
        #             if len(reached) >= required_count:
        #                 return True
        return False
