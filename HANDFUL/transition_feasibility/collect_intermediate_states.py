from dataclasses import dataclass
import tyro
import os, sys
import torch
import numpy as np
import tqdm
import gymnasium as gym
from collections import defaultdict
from typing import Optional



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode

from train import Actor

@dataclass
class CollectionArgs:
    """Arguments for intermediate state collection."""
    checkpoint: str
    """Path to trained model checkpoint"""
    env_id: str = "xArm7-v1-pick-randomized"
    """The environment ID"""
    control_mode: str = "pd_joint_delta_pos"
    """Control mode for the environment"""
    num_states_to_save: int = 16384
    """Number of successful episodes to collect"""
    num_steps_to_save: int = 10
    """Number of final steps to save per episode"""
    num_envs: int = 512
    """Number of parallel environments for main collection"""
    num_envs_viz: int = 10
    """Number of parallel environments for visualization collection"""
    num_states_viz: int = 10
    """Number of states to save with images for visualization"""
    save_images: bool = True
    """Whether to save images for visualization collection"""
    save_videos: bool = True
    """Whether to save videos"""
    robot_uids: str = "xarm7_leap_right"
    """Robot UID"""
    export_glb: bool = True
    """Whether to export a GLB alongside each final image"""

    # randomized picking specifics
    finger_selection: Optional[str] = None
    """Finger selection: comma-separated list like '0,1,2,3'"""
    palm_use: bool = False
    """Whether to use palm contact"""
    num_active_fingers: int = 2
    """Number of active fingers (1-3)"""

    min_episodes_for_cutoff: int = 5000
    """Minimum number of episodes run before checking for success rate cutoff"""
    success_rate_cutoff: float = 15.0
    """Success rate threshold (%) below which state collection is terminated"""


# ---------------------------------------------------------------------------
# GLB export helper
# ---------------------------------------------------------------------------

def _get_scene_as_trimesh(env_unwrapped):
    import trimesh
    import sapien.render

    scene = trimesh.scene.Scene()

    def _add_objs(render_shapes, pose_mat, name_prefix, extra_scale=1.0):
        for rs_idx, render_shape in enumerate(render_shapes):
            scale = np.array(render_shape.scale) if hasattr(render_shape, "scale") else np.ones(3)
            scale = scale * extra_scale

            # Extract material color
            color = None
            try:
                mat = render_shape.material
                base_color = mat.base_color 
                color = (np.array(base_color) * 255).astype(np.uint8)
            except Exception:
                pass

            # local_pose is on the render_shape, not on individual parts
            if hasattr(render_shape, "local_pose"):
                local_mat = np.array(render_shape.local_pose.to_transformation_matrix())
                world_mat = pose_mat @ local_mat
            else:
                world_mat = pose_mat

            for part_idx, part in enumerate(render_shape.parts):
                verts = np.array(part.vertices) * scale
                faces = np.array(part.triangles) 
                if verts.shape[0] == 0 or faces.shape[0] == 0:
                    continue

                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                if color is not None:
                    mesh.visual.face_colors = np.tile(color, (len(faces), 1))
                mesh.apply_transform(world_mat)
                scene.add_geometry(mesh, node_name=f"{name_prefix}_rs{rs_idx}_p{part_idx}")

    # --- Actors ---
    for actor, extra_scale in [
        #(env_unwrapped.table_scene.table, 1.0),
        (env_unwrapped.cube,              0.025),
        ]:
        obj = actor._objs[0]
        pose_mat = actor.pose.to_transformation_matrix()[0].cpu().numpy()
        render_comp = obj.find_component_by_type(sapien.render.RenderBodyComponent)
        if render_comp is not None:
            _add_objs(render_comp.render_shapes, pose_mat, obj.name, extra_scale=extra_scale)
    
    # --- Robot links ---
    for link in env_unwrapped.agent.robot.links:
        obj = link._objs[0]
        pose_mat = link.pose.to_transformation_matrix()[0].cpu().numpy()
        render_comp = obj.entity.find_component_by_type(sapien.render.RenderBodyComponent)
        if render_comp is not None:
            _add_objs(render_comp.render_shapes, pose_mat, link.name)

    z_to_y_up = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    scene.apply_transform(z_to_y_up)

    return scene


# ---------------------------------------------------------------------------
# Main collection function
# ---------------------------------------------------------------------------

def collect_intermediate_states(
    env_id: str,
    checkpoint: str,
    num_envs: int,
    control_mode: str,
    saved_states_dir: str,
    saved_states_name: str,
    num_states_to_save: int,
    num_steps_to_save: int,
    save_images: bool = False,
    export_glb: bool = False,
    video_output_dir: Optional[str] = None,
    finger_selection: Optional[str] = None,
    palm_use: bool = False,
    num_active_fingers: int = 2,
    robot_uids: str = "xarm7_leap_right",
    min_episodes_for_cutoff: int = 5000,
    success_rate_cutoff: float = 15.0,
):
    """
    Collects intermediate states (last N steps) from successful episodes.
    """

    rng = np.random.default_rng() 
    
    os.makedirs(saved_states_dir, exist_ok=True)
    
    if video_output_dir is not None:
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"Videos will be saved to: {video_output_dir}")
    
    if save_images:
        images_dir = os.path.join(saved_states_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        print(f"Images will be saved to: {images_dir}")
    
    # Setup environment
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="gpu",
        control_mode=control_mode,
        robot_uids=robot_uids,
    )

    # Whole-hand environments (like xArm7-v1-pick-all) do not support finger selection parameters
    is_whole_hand = env_id.endswith("-all") or "whole-hand" in env_id or "whole_hand" in env_id
    if not is_whole_hand and finger_selection is not None:
        if ',' in finger_selection:
            env_kwargs["finger_selection"] = [int(x) for x in finger_selection.split(',')]
        else:
            env_kwargs["finger_selection"] = finger_selection
        env_kwargs["num_active_fingers"] = num_active_fingers
        env_kwargs["palm_use"] = palm_use
    
    env = gym.make(
        env_id,
        num_envs=num_envs,
        reconfiguration_freq=1,
        human_render_camera_configs=dict(shader_pack="default"),
        **env_kwargs
    )

    num_eval_steps = gym_utils.find_max_episode_steps_value(env)
    
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    
    if video_output_dir is not None:
        env = RecordEpisode(
            env,
            output_dir=video_output_dir,
            save_trajectory=False,
            save_video=True,
            video_fps=30,
            max_steps_per_video=num_eval_steps,
        )

    env = ManiSkillVectorEnv(
        env,
        num_envs,
        ignore_terminations=True,
        record_metrics=True
    )
    
    print(f"Loading policy from {checkpoint}")
    actor = Actor(env).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()
    
    episodes_saved = 0
    total_episodes_run = 0
    all_successful_episodes = []
    
    pbar_states = tqdm.tqdm(total=num_states_to_save, desc="Collecting states")
    
    while episodes_saved < num_states_to_save:
        obs, info = env.reset(seed=rng.integers(low=0, high=1000))
        
        episode_data = defaultdict(list)
        successful_envs_in_episode = [False] * num_envs
        
        for t in range(1, num_eval_steps):
            with torch.no_grad():
                actions = actor.get_eval_action(obs)
                obs, rewards, terminated, truncated, infos = env.step(actions)
            
            # Retrieve and log the state during the final N steps of the episode.
            # These steps correspond to when the grasp is fully stabilized.
            if t >= num_eval_steps - num_steps_to_save:
                
                # Fetch full simulation state dictionary from SAPIEN/ManiSkill
                current_states = env.unwrapped.get_state_dict()
                tcp_poses = env.unwrapped.agent.tcp_pose.raw_pose
                cube_half_sizes = env.unwrapped.cube_half_sizes_per_env
                
                # If custom finger selection is used, extract current active/inactive finger indices
                active_finger_indices = None
                inactive_finger_indices = None
                if hasattr(env.unwrapped, "active_finger_indices"):
                    active_finger_indices = env.unwrapped.active_finger_indices
                    inactive_finger_indices = env.unwrapped.inactive_finger_indices
                
                # Extract individual environment states from batched GPU tensors
                for i in range(num_envs):
                    env_state_at_step_t = {}
                    
                    # Extract positions/orientations/velocities of free rigid actors (e.g. the cube)
                    actors_states = current_states.get('actors', {})
                    for object_name, tensor in actors_states.items():
                        env_state_at_step_t[object_name] = tensor[i].clone()
                    
                    # Extract articulation states (e.g. robot joint positions and velocities)
                    articulations_states = current_states.get('articulations', {})
                    for object_name, tensor in articulations_states.items():
                        env_state_at_step_t[object_name] = tensor[i].clone()
                    
                    # Store tool center point (TCP) pose and cube geometry context
                    env_state_at_step_t['tcp_pose'] = tcp_poses[i].clone()
                    env_state_at_step_t['cube_half_sizes'] = cube_half_sizes[i].clone()
                    
                    if active_finger_indices is not None:
                        env_state_at_step_t['active_finger_indices'] = active_finger_indices[i].clone()
                        env_state_at_step_t['inactive_finger_indices'] = inactive_finger_indices[i].clone()
                    
                    episode_data[i].append(env_state_at_step_t)
            
            # At the final step of the episode, determine which parallel envs were successful.
            # Only successful episodes are kept to populate the intermediate transition state pool.
            if t == num_eval_steps - 1:
                top_level_success = infos.get("success", None)
                final_info_success = infos.get("final_info", {}).get("success", None)
                
                if top_level_success is not None and final_info_success is not None:
                    grasped_mask = top_level_success | final_info_success
                elif top_level_success is not None:
                    grasped_mask = top_level_success
                elif final_info_success is not None:
                    grasped_mask = final_info_success
                else:
                    grasped_mask = None
                
                if grasped_mask is not None:
                    for i in range(num_envs):
                        if grasped_mask[i]:
                            successful_envs_in_episode[i] = True
                
                if save_images:
                    final_images = env.render()

                final_trimesh_scene = None
                if export_glb:
                    try:
                        final_trimesh_scene = _get_scene_as_trimesh(env.unwrapped)
                    except Exception as exc:
                        print(f"  [warn] GLB capture failed: {exc}")
        
        successful_in_this_batch = 0
        
        for i, has_succeeded in enumerate(successful_envs_in_episode):
            if has_succeeded and episodes_saved < num_states_to_save:
                episode_trajectory = episode_data[i]
                
                if save_images:
                    img_data = final_images[i]
                    img_array = img_data.cpu().numpy() if isinstance(img_data, torch.Tensor) else img_data
                    img = Image.fromarray(img_array.astype(np.uint8))
                    img_filename = f"episode_{episodes_saved}_final.png"
                    img.save(os.path.join(images_dir, img_filename))
                    episode_trajectory[-1]['image_path'] = img_filename

                if export_glb and final_trimesh_scene is not None:
                    glb_filename = f"{env_id}_episode_{episodes_saved}_final.glb"
                    final_trimesh_scene.export(os.path.join(images_dir, glb_filename))
                    episode_trajectory[-1]['glb_path'] = glb_filename
                
                all_successful_episodes.append(episode_trajectory)
                episodes_saved += 1
                successful_in_this_batch += 1
                pbar_states.update(1)
        
        total_episodes_run += num_envs
        
        batch_success_rate = (successful_in_this_batch / num_envs) * 100
        overall_success_rate = (episodes_saved / total_episodes_run) * 100
        print(f"Batch: {successful_in_this_batch}/{num_envs} successful ({batch_success_rate:.1f}%) | "
              f"Overall: {episodes_saved}/{total_episodes_run} ({overall_success_rate:.1f}%)")

        if total_episodes_run >= min_episodes_for_cutoff and overall_success_rate < success_rate_cutoff:
            print(f"WARNING: Success rate is too low ({overall_success_rate:.2f}% < {success_rate_cutoff}% "
                  f"after {total_episodes_run} episodes). Terminating state collection.")
            pbar_states.close()
            env.close()
            return False
    
    output_path = os.path.join(saved_states_dir, f"{saved_states_name}.pt")
    torch.save(all_successful_episodes, output_path)
    pbar_states.close()
    
    print("\n" + "="*60)
    print("STATE COLLECTION SUMMARY")
    print("="*60)
    print(f"Total episodes run: {total_episodes_run}")
    print(f"Successful episodes: {episodes_saved}")
    print(f"Failed episodes: {total_episodes_run - episodes_saved}")
    print(f"Overall success rate: {(episodes_saved / total_episodes_run * 100):.2f}%")
    print(f"Saved to: {output_path}")
    print("="*60 + "\n")
    
    env.close()
    print("State collection complete.")
    
    if save_images:
        print(f"Saved {episodes_saved} final images to {images_dir}")
        
    return True


if __name__ == "__main__":

    args = tyro.cli(CollectionArgs)

    if args.checkpoint is None or not os.path.exists(args.checkpoint):
        raise ValueError(f"Invalid checkpoint path: {args.checkpoint}")

    
    # Derive directory name from checkpoint
    dir_name = os.path.dirname(args.checkpoint)
    run_name = os.path.basename(dir_name)
    saved_states_dir = f"intermediate_states/{run_name}"
    video_output_dir = f"{saved_states_dir}/videos" if args.save_videos else None
    
    # Example 2: Collect main dataset without images (large batch) first
    print("\nCollecting main dataset...")
    success = collect_intermediate_states(
        env_id=args.env_id,
        checkpoint=args.checkpoint,
        num_envs=args.num_envs,
        control_mode=args.control_mode,
        saved_states_dir=saved_states_dir,
        saved_states_name="grasp_to_push",
        num_states_to_save=args.num_states_to_save,
        num_steps_to_save=args.num_steps_to_save,
        save_images=False,
        finger_selection=args.finger_selection,
        palm_use=args.palm_use,
        num_active_fingers=args.num_active_fingers,
        robot_uids=args.robot_uids,
        min_episodes_for_cutoff=args.min_episodes_for_cutoff,
        success_rate_cutoff=args.success_rate_cutoff,
    )
    if not success:
        print("Main state collection failed/cutoff. Exiting.")
        sys.exit(1)

    # Collect with images for visualization (small batch) only if main succeeds
    if args.save_images and args.num_states_viz > 0:
        print("\nCollecting states with images for visualization...")
        success = collect_intermediate_states(
            env_id=args.env_id,
            checkpoint=args.checkpoint,
            num_envs=args.num_envs_viz,
            control_mode=args.control_mode,
            saved_states_dir=saved_states_dir,
            saved_states_name="grasp_to_push_img",
            num_states_to_save=args.num_states_viz,
            num_steps_to_save=args.num_steps_to_save,
            save_images=True,
            export_glb=args.export_glb,
            video_output_dir=video_output_dir,
            finger_selection=args.finger_selection,
            palm_use=args.palm_use,
            num_active_fingers=args.num_active_fingers,
            robot_uids=args.robot_uids,
            min_episodes_for_cutoff=min(args.min_episodes_for_cutoff, 10 * args.num_envs_viz),
            success_rate_cutoff=args.success_rate_cutoff,
        )
        if not success:
            print("Visualization state collection failed/cutoff. Exiting.")
            sys.exit(1)
    
    # Visualization (run in a separate subprocess to avoid Vulkan/Matplotlib segfault on exit)
    try:
        print("\nAnalyzing state diversity...")
        import subprocess
        # Try importing plotting module locally first to ensure it exists
        try:
            from transition_feasibility import intermediate_variance_plotting
            plotting_import_path = "from transition_feasibility import intermediate_variance_plotting"
        except ImportError:
            import intermediate_variance_plotting
            plotting_import_path = "import intermediate_variance_plotting"
        
        cmd = [
            "python3", "-c",
            f"import matplotlib; matplotlib.use('Agg'); import torch; {plotting_import_path}; "
            f"saved_states = torch.load('{saved_states_dir}/grasp_to_push.pt'); "
            f"intermediate_variance_plotting.visualize_block_positions_in_tcp_frame(saved_states, '{saved_states_dir}')"
        ]
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"  [warn] Diversity visualization failed: {e}")