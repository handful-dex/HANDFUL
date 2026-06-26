import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import gymnasium as gym
from envs.tasks.xarm7_allegron_env import XArm7TableTop
import h5py
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation as R
# Resolve path relative to this script to make it agnostic to folder renaming
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HDF5_FILEPATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "leap_robot_teleop.hdf5"))

class TrajectoryReplayer:
    def __init__(self, hdf5_filepath=HDF5_FILEPATH):
        if not os.path.exists(hdf5_filepath):
            raise FileNotFoundError(f"HDF5 file not found at: {hdf5_filepath}")
        
        self.hdf5_filepath = hdf5_filepath
        self.arm = None
        self.hand = None
        
        self.ee_origin_default_offset = [0, 0, 0.12] # Must match teleop recording logic

        print(f"Trajectory Replayer initialized. Will read from: {self.hdf5_filepath}")

    def load_trajectory(self):
        episodes = []
        with h5py.File(self.hdf5_filepath, 'r') as f:
            for key in f.keys():
                if key.startswith('episode_') and key[8:].isdigit():
                    episode_id = int(key[8:])
                    description = f[key]['metadata']['task_description'][()] if 'task_description' in f[key]['metadata'] else "N/A"
                    success = f[key]['metadata']['success'][()] if 'success' in f[key]['metadata'] else "N/A"
                    duration = f[key]['metadata']['episode_duration_seconds'][()] if 'episode_duration_seconds' in f[key]['metadata'] else "N/A"
                    
                    # Check for video data
                    has_rgb = 'camera_data/rgb_images' in f[key]
                    has_depth = 'camera_data/depth_images' in f[key]

                    episodes.append({
                        'id': episode_id,
                        'name': key,
                        'description': description.decode('utf-8') if isinstance(description, bytes) else description, # Decode if bytes
                        'success': success,
                        'duration': duration,
                        'has_video': has_rgb or has_depth
                    })
        episodes.sort(key=lambda x: x['id'])
        if not episodes:
            print(f"No episodes found in {self.hdf5_filepath}. Did you record any?")

        return episodes
    
    def load_episode(self, episode_id):
        """Loads data for a specific episode."""
        episode_data = {}
        try:
            with h5py.File(self.hdf5_filepath, 'r') as f:
                episode_group_name = f'episode_{episode_id:04d}'
                if episode_group_name not in f:
                    raise ValueError(f"Episode {episode_id} not found in HDF5 file.")
                
                episode_group = f[episode_group_name]
                
                # Load robot data
                robot_data_group = episode_group['robot_data']
                episode_data['timestamps'] = robot_data_group['timestamps'][()]
                episode_data['arm_joint_positions'] = robot_data_group['arm_joint_positions'][()]
                episode_data['arm_end_effector_pose'] = robot_data_group['arm_end_effector_pose'][()]
                episode_data['hand_joint_positions'] = robot_data_group['hand_joint_positions'][()]
                episode_data['teleop_arm_commands'] = robot_data_group['teleop_arm_commands'][()] # The RPY commands
                episode_data['teleop_hand_commands'] = robot_data_group['teleop_hand_commands'][()]

                # Load camera data if available
                if 'camera_data' in episode_group:
                    camera_data_group = episode_group['camera_data']
                    if 'rgb_images' in camera_data_group:
                        episode_data['rgb_images'] = camera_data_group['rgb_images'][()]
                    if 'depth_images' in camera_data_group:
                        episode_data['depth_images'] = camera_data_group['depth_images'][()]
                    if 'metadata' in episode_group and 'camera_intrinsics' in episode_group['metadata']:
                        episode_data['camera_intrinsics'] = episode_group['metadata']['camera_intrinsics'][()]
                    if 'metadata' in episode_group and 'camera_extrinsics' in episode_group['metadata']:
                        episode_data['camera_extrinsics'] = episode_group['metadata']['camera_extrinsics'][()]

                print(f"Successfully loaded episode {episode_id}.")
                return episode_data

        except Exception as e:
            print(f"Error loading episode {episode_id}: {e}")
            return None
    def parse_episode_data(self, episode_data, arm_control_mode="ee_pose", hand_control_model="joint_pos"):
        """Parses the episode data to extract arm and hand commands."""
        if 'teleop_arm_commands' not in episode_data or 'teleop_hand_commands' not in episode_data:
            raise ValueError("Episode data must contain 'teleop_arm_commands' and 'teleop_hand_commands'.")

        timestamps = episode_data['timestamps']
        arm_commands_rpy = episode_data['teleop_arm_commands'] # Use the RPY commands as recorded
        hand_commands = episode_data['teleop_hand_commands']

        # === prepare the action batch ===
        action_batch = []
        for i in range(len(timestamps)):

            # Get the target for the current frame
            current_arm_target_rpy = arm_commands_rpy[i]
            current_hand_target = hand_commands[i]

            # Apply origin offset (specific to xArm)
            current_arm_targets_mm = (np.array(current_arm_target_rpy[0:3])).tolist()
            current_target_rot = R.from_euler('xyz', current_arm_target_rpy[3:6])
            current_arm_targets_mm_offset = current_arm_targets_mm + current_target_rot.apply(np.array(replayer.ee_origin_default_offset))

            # Concatenate arm and hand commands
            action = np.concatenate((current_arm_targets_mm_offset*0.001, current_arm_target_rpy[3:6], current_hand_target))
            action_batch.append(action)

        # == Calculate the delta actions ==
        action_batch = np.array(action_batch)

        if arm_control_mode == "ee_pose":
            return action_batch
        elif arm_control_mode == "ee_delta_pose":
            # Calculate delta actions for end-effector pose
            delta_actions = np.zeros_like(action_batch)
            delta_actions[0] = action_batch[0]  # First action is the same
            for i in range(1, len(action_batch)):
                delta_actions[i, :6] = action_batch[i, :6] - action_batch[i-1, :6]
                delta_actions[i, 6:] = action_batch[i, 6:]
            return delta_actions
        
if __name__ == "__main__":
    env = gym.make("xArm7-v1", 
                   robot_uids="xarm7_leap_right", 
                   num_envs=1,
                   obs_mode="state", 
                   control_mode="pd_ee_pose",  #"pd_ee_delta_pose", "pd_joint_pos", "pd_ee_pos"
                   render_mode="human", 
                   viewer_camera_configs=dict(fov=60))
    # Parse the parameters when running the code of episode number


    #TODO: xarm7 lack convex stl files, so we cannot use the motion planner yet
    # planner = XArm7DexhandPlanningSolver(
    #     env,
    #     debug=False,
    #     vis=True,
    #     base_pose=env.unwrapped.agent.robot.pose,
    #     visualize_target_grasp_pose=True,
    #     print_env_info=True
    # )
    obs = env.reset()

    # Optional: set camera or viewer position
    viewer = env.viewer
    viewer.set_camera_xyz(x=0.5, y=0.5, z=0.5)
    viewer.set_camera_rpy(r=0, p=-np.pi/6, y=4*np.pi/6)
    
    replayer = TrajectoryReplayer(HDF5_FILEPATH)
    episodes = replayer.load_trajectory()
    print(f"Found {len(episodes)} episodes in the dataset.")
    print("Episodes:")
    for ep in episodes:
        print(f"ID: {ep['id']}, Name: {ep['name']}, Success: {ep['success']}, Duration: {ep['duration']}s, Description: {ep['description']}, Has Video: {ep['has_video']}")
    
    episode_id = 23
    print(f"Loading episode {episode_id}...")
    episode_data = replayer.load_episode(episode_id)
    action_batch = replayer.parse_episode_data(episode_data, arm_control_mode = "ee_pose", hand_control_model= "joint_pos")
    # Print the steps of the action batch
    print(f"Loaded {len(action_batch)} actions for episode {episode_id}.")

    # === Replay the trajectory ===
    print(f"Replaying episode {episode_id} with {len(action_batch)} actions.")
    ee_position = env.cube.pose.p[0].tolist()
    
    for i, action in enumerate(action_batch):
        print(f"Step {i+1}/{len(action_batch)}: Action = {action}")
        action = np.zeros(env.action_space.shape)
        action[:6] = [
            ee_position[0] - 0.12 + 0.65, #+ 0.002 * i,  # x position
            ee_position[1] - 0.04,  # y position
            ee_position[2] + 0.13, # + 0.2,  #+ 0.3 - 0.001 *i,  # z position
            0.0,
            3.14,
            0.0,
        ]
        action[6:] = action_batch[i][6:]#0.0  # Reset the hand action to zero for the first action
        obs, reward, terminated, truncated, info = env.step(action)
        # Render at a specific viewer camera
        env.render()  # Render the environment

    # Testing grasping
    for i in range(100):
        action = np.zeros(env.action_space.shape)
        action[:6] = [
            ee_position[0] - 0.12 + 0.65,  # x position
            ee_position[1] - 0.04,  # y position
            ee_position[2] + 0.13 + 0.1,  # z position
            0.0,
            3.14,
            0.0,
        ]
        action[6:] = action_batch[-1][6:]  # Close the hand
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
    env.close()