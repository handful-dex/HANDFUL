import torch
import gymnasium as gym
import mani_skill.envs
import numpy as np
from PIL import Image

from envs.tasks.xarm7_leap_pick_env import XArm7TableTop
from envs.tasks.xarm7_leap_push_env import XArm7TableTopPush
from envs.tasks.xarm7_press_button import XArm7TableTopPress
from envs.tasks.xarm7_pick_randomized import XArm7TableTopPickRandomized
from envs.tasks.xarm7_pick_all import XArm7TableTopPickAll
from envs.tasks.xarm7_two_pick import XArm7TableTopTwoPick
from envs.tasks.xarm7_twist import XArm7TableTopKnobTwist
from envs.tasks.xarm7_open_drawer import XArm7CabinetDrawerEnv, XArm7CabinetDoorEnv

from envs.tasks.unified_environments.xarm7_two_pick_unified import XArm7TableTopTwoPickUnified
from envs.tasks.unified_environments.xarm7_push_unified import XArm7TableTopPushUnified
from envs.tasks.unified_environments.xarm7_open_drawer_unified import XArm7CabinetDrawerEnvUnified
from envs.tasks.unified_environments.xarm7_press_button_unified import XArm7TableTopPressUnified
from envs.tasks.unified_environments.xarm7_twist_unified import XArm7TableTopKnobTwistUnified

from envs.tasks.whole_hand_environments.xarm7_push_whole_hand import XArm7TableTopPushWholeHand
from envs.tasks.whole_hand_environments.xarm7_open_drawer_whole_hand import XArm7CabinetDrawerWholeHand
from envs.tasks.whole_hand_environments.xarm7_press_button_whole_hand import XArm7TableTopPressWholeHand
from envs.tasks.whole_hand_environments.xarm7_twist_whole_hand import XArm7TableTopKnobTwistWholeHand
from envs.tasks.whole_hand_environments.xarm7_two_pick_whole_hand import XArm7TableTopTwoPickWholeHand


from mani_skill.utils import gym_utils
from mani_skill.utils.structs.pose import Pose


num_envs = 1  # Use 1 env for cleaner blending, or pick one env's frame
blended = None

env_kwargs = dict(
    robot_uids="xarm7_leap_right",
    sim_backend="gpu",
    difficulty=1,
)

env = gym.make(
    "xArm7-v1-cabinet-drawer",
    num_envs=num_envs,
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    render_mode="rgb_array",  # <-- capture frames instead of displaying
    human_render_camera_configs=dict(shader_pack="default"),  # <-- add this
    viewer_camera_configs=dict(fov=60),
    **env_kwargs
)

num_resets = 5
frames = []

obs, _ = env.reset(seed=0)

# Move blocks far away so they're out of frame for background capture
far_pose = Pose.create_from_pq(
    torch.tensor([[999, 999, 999]], dtype=torch.float32).expand(num_envs, -1),
    torch.tensor([[1, 0, 0, 0]], dtype=torch.float32).expand(num_envs, -1)
)
# env.unwrapped.cube.set_pose(far_pose)
# env.unwrapped.push_block.set_pose(far_pose)
# env.unwrapped.push_goal_site.set_pose(far_pose)

# env.unwrapped.valve.set_pose(far_pose)
# env.unwrapped.valve_link.set_pose(far_pose)
# env.unwrapped.wall.set_pose(far_pose)

env.unwrapped.cabinet.set_pose(far_pose)


if env.unwrapped.gpu_sim_enabled:
    env.unwrapped.scene._gpu_apply_all()
    env.unwrapped.scene.px.gpu_update_articulation_kinematics()
    env.unwrapped.scene.px.step()
    env.unwrapped.scene._gpu_fetch_all()
# env.unwrapped.box.set_pose(far_pose)

# Need to step physics/render to apply
env.unwrapped.scene.step()  # or just render directly

background = env.render()[0]
background = (background.cpu().numpy() if isinstance(background, torch.Tensor) else background).astype(np.float32)

for i in range(num_resets):
    obs, _ = env.reset(seed=2*i)
    final_images = env.render()
    
    for j in range(len(final_images)):
        img_data = final_images[j]
        frame = img_data.cpu().numpy() if isinstance(img_data, torch.Tensor) else img_data
        frames.append(frame.astype(np.float32))


# Composite per-frame first, then max blend
composited_frames = []

dim_background = background * 0.8

for i, frame in enumerate(frames):
    diff = np.abs(frame - background).sum(axis=-1)  # (H, W)
    mask = (diff > 10).astype(np.float32)[..., np.newaxis]  # (H, W, 1)
    
    composited = mask * frame + (1 - mask) * dim_background
    composited_frames.append(composited)

alpha = 1.0  # 1.0 = fully opaque blocks, 0.0 = invisible

fg_accumulator = np.full_like(background, -np.inf)

for frame in frames:
    diff = np.abs(frame - background).sum(axis=-1)
    mask = (diff > 20)
    fg_accumulator[mask] = np.maximum(fg_accumulator[mask], frame[mask])

never_fg = fg_accumulator[..., 0] == -np.inf
result = fg_accumulator.copy()

# Blend foreground with background at alpha
fg_mask = ~never_fg
result[fg_mask] = alpha * fg_accumulator[fg_mask] + (1 - alpha) * background[fg_mask]

# Pure background where no foreground was ever seen
result[never_fg] = background[never_fg]

result_img = Image.fromarray(result.astype(np.uint8))
result_img.save("randomization_blend_cabinet1.png")
result_img.show()

env.close()