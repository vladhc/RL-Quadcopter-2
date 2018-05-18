import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6 + 6 * 3 # 9 for 6x2 for velocity arr, 6x1 for acc arr
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        # For states preprocessing
        init_pose = self.sim.pose
        r = self.target_pos - init_pose[:3]
        self.initial_distance = np.linalg.norm(r)

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        dist_from_target = np.linalg.norm(self.target_pos - self.sim.pose[:3])
        dist_reward = 1.0 - np.tanh(dist_from_target / 3.0)
        time_reward = 0.1
        reward = time_reward + dist_reward
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(np.copy(self.sim.pose))
            # Early exit, so we won't pollute pose_all with non-valid states
            if done:
                break

        next_state = self._preprocess_pose(pose_all)
        return next_state, reward, done

    def _preprocess_pose(self, pose_all):
        while len(pose_all) < self.action_repeat:
            pose_all.append(np.copy(pose_all[-1]))
        velocity1 = pose_all[1] - pose_all[0]
        velocity2 = pose_all[2] - pose_all[1]
        acc = velocity2 - velocity1
        for i, pose in enumerate(pose_all):
            # (0,0,0) is a target_pos
            coords_relative_to_target = self.target_pos - pose[:3]
            # normalize angles
            angles = np.copy(pose[3:])
            angles /= (2 * np.pi)
            state = np.copy(pose)
            state[:3] = coords_relative_to_target
            state[3:] = angles
            pose_all[i] = state
        pose_all.append(velocity1)
        pose_all.append(velocity2)
        pose_all.append(acc)
        state = np.concatenate(pose_all)
        return state

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        pose_all = [self.sim.pose]
        return self._preprocess_pose(pose_all)

    def render(self):
        if not hasattr(self, 'scene'):
            from scene import Scene
            self.scene = Scene(self.sim, self.target_pos)
        self.scene.update(self.sim.pose)
