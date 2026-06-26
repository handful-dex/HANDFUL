import torch 

"""
PickCube-v1 is a basic/common task which defaults to using the panda robot. It is also used as a testing task to check whether a robot with manipulation
capabilities can be simulated and trained properly. The configs below set the pick cube task differently to ensure the cube is within reach of the robot tested
and the camera angles are reasonable.
"""
CUBE_VARIATION_DIMENSIONS = [
                            [0.025, 0.025, 0.025],
                            [0.0275, 0.0275, 0.0275],
                            [0.0275, 0.03, 0.03],
                            [0.03, 0.03, 0.03],
                            [0.025, 0.0275, 0.03]
                            ]

PUSH_BLOCK_VARIATION_DIMENSIONS = [[0.025, 0.025, 0.04]]

MASS_VARIATIONS = [0.5, 1.5]
FRICTION_VARIATIONS = [0.1, 0.5]
RESTITUTION_VARIATIONS = [0.0, 0.1]



PICK_CUBE_CONFIGS = {
    "xarm7_leap_right": {
        "cube_variation_dimensions": CUBE_VARIATION_DIMENSIONS,
        "goal_thresh": 0.03,
        "cube_spawn_half_size": 0.05,
        "cube_spawn_center": (-0.1, -0.0),
        "max_goal_height": 0.25,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.5, -0.5, 0.6],
        "human_cam_target_pos": [-0.1, 0.0, 0.35],
    },

}

PICK_REWARD_CONFIGS = {
    "sac": {
        "reach_weight": 2.0,
        "finger_dist_weight": 7.5,
        "grasp_weight": 2.5,
        "inactive_finger_penalty_weight": 1.0,
        "place_weight": 40.0,
        "static_penalty_weight": 1.5,
        "collision_penalty_weight": 1.0,
    },
}

PICK_PUSH_CONFIGS = {
    "xarm7_leap_right": {
        "cube_half_size": 0.025,
        "cube_variation_dimensions": CUBE_VARIATION_DIMENSIONS,
        "goal_thresh": 0.03,
        "rot_thresh": 0.5,
        "cube_spawn_half_size": 0.05,
        "push_block_spawn_center": (-0.1, 0.25),
        "push_block_half_sizes": [0.045, 0.045, 0.1],
        "push_block_rot": [1, 0, 0, 0],
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.25, 0.5],
        "human_cam_target_pos": [-0.1, 0.25, 0.15],
    },

}

PICK_PUSH_REWARD_CONFIGS = {
    "sac": {
        "grasp_weight": 7.5,
        "reach_weight": 3.0,
        "push_weight": 3.0,
        "place_weight": 20.0,
        "rot_weight": 5.0,
        "inactive_finger_penalty_weight": 1.0,
        "static_penalty_weight": 1.0,
        "collision_penalty_weight": 1.0,
    },
}

PICK_PRESS_CONFIGS = {
        "xarm7_leap_right": {
        "cube_variation_dimensions": CUBE_VARIATION_DIMENSIONS,
        "goal_thresh": 0.04,
        "push_block_spawn_half_size": 0.25,
        "push_block_spawn_rot_noise": 0.261799,
        "push_block_spawn_center": [0.3, 0.0, 0.35],
        "push_block_half_sizes": [0.025, 0.025, 0.01],
        "push_block_rot": [0, 0.7071068, 0, 0.7071068],
        "box_with_hole_radii_and_depth": [0.04, 0.1, 0.1],
        "max_goal_height": 0.25,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [-0.7, -0.7, 0.5],
        "human_cam_target_pos": [0.1, 0.0, 0.35],
    },
}

PICK_PRESS_REWARD_CONFIGS = {
    "sac": {
        "reach_weight": 7.5,
        "grasp_weight": 2.0,
        "inactive_finger_penalty_weight": 1.0,
        "static_penalty_weight": 1.0,
        "collision_penalty_weight": 1.0,
    },
}

TWO_PICK_CONFIGS = {
    "xarm7_leap_right": {
    "cube_half_sizes": [0.025, 0.025, 0.03],
    "cube_variation_dimensions": CUBE_VARIATION_DIMENSIONS,
    "push_block_variation_dimensions": PUSH_BLOCK_VARIATION_DIMENSIONS,
    "cube_spawn_half_size": 0.15,
    "goal_thresh": 0.03,
    "push_block_spawn_center": [-0.1, -0.3, 0.5],
    "push_block_half_sizes": [0.025, 0.025, 0.035],
    "push_block_rot": [1, 0, 0, 0],
    "max_goal_height": 0.3,
    "sensor_cam_eye_pos": [0.3, 0, 0.6],
    "sensor_cam_target_pos": [0.0, 0, 0.1],
    "human_cam_eye_pos": [0.5, -0.15, 0.5],
    "human_cam_target_pos": [0.0, -0.15, 0.25],
    },
}

TWO_PICK_REWARD_CONFIGS = {
    "sac": {
        "grasp_reward": 7.5,
        "reach_weight": 3.0,
        "finger_dist_weight": 7.5,
        "push_block_grasp_weight": 2.5,
        "inactive_finger_penalty_weight": 1.0,
        "height_weight": 20.0,
        "place_weight": 30.0,
        "static_penalty_weight": 0.5,
        "collision_penalty_weight": 0.2,
    },
}

KNOB_TWIST_CONFIGS = {
    "xarm7_leap_right": {
        "cube_half_sizes": [0.025, 0.025, 0.03],
        "cube_variation_dimensions": CUBE_VARIATION_DIMENSIONS,
        "goal_thresh": 0.03,
        "valve_spawn_pos": [0.2, 0.0, 0.35],
        "valve_spawn_noise": 0.15,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [-0.2, -0.7, 0.5],
        "human_cam_target_pos": [0.1, 0.0, 0.35],
    }
}

KNOB_TWIST_REWARD_CONFIGS = {
    "sac": {
        "grasp_weight": 7.5,
        "inactive_finger_penalty_weight": 1.0,
        "valve_reach_weight": 3.0,
        "fingertip_reach_weight": 7.5,
        "velocity_weight": 3.0,
        "motion_weight": 10.0,
        "static_penalty_weight": 0.5,
        "collision_penalty_weight": 2.0,
    },
}

CABINET_CONFIGS = {
    "xarm7_leap_right": {
        "cube_variation_dimensions": CUBE_VARIATION_DIMENSIONS,
        "goal_thresh": 0.03,
        "cabinet_radius": 0.8,
        "cabinet_angle_range": [-torch.pi/4, torch.pi/4],
        "cabinet_spawn_noise": 0.05,
        "cabinet_spawn_rot_noise": 0.261799,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [-0.2, -0.7, 0.5],
        "human_cam_target_pos": [0.1, 0.0, 0.35],
    },
}

CABINET_REWARD_CONFIGS = {
    "sac": {
        "grasp_weight": 5.0,
        "reach_weight": 1.0,
        "finger_dist_weight": 5.0,
        "open_weight": 15.0,
        "inactive_finger_penalty_weight": 1.0,
        "static_penalty_weight": 0.1,
        "collision_penalty_weight": 3.0,
    },
}