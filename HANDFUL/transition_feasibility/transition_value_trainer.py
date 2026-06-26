import os
import random
import torch
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, List, Tuple

from terminal_value_function import GraspInsertTValue

@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower

class TValue_Trainer():
    """Trainer for T-Value functions."""

    def __init__(self, 
                 success_file_path: Optional[str] = None,
                 failure_file_path: Optional[str] = None,
                 model_name: str = "default_model",
                 state_keys: Optional[List[str]] = None,
                 use_all_states: bool = True,
                 validation_split: float = 0.1,
                 success_episodes: Optional[List] = None,
                 failure_episodes: Optional[List] = None) -> None:
        """
        Initialize the TValue trainer.
        
        Args:
            success_file_path: Path to .pt file with successful episodes (or None if using success_episodes)
            failure_file_path: Path to .pt file with failure episodes (or None if using failure_episodes)
            model_name: Name for saving the model
            state_keys: List of keys to use as input
            use_all_states: If True, use all available state keys
            validation_split: Fraction of data for validation
            success_episodes: Pre-loaded success episodes (alternative to success_file_path)
            failure_episodes: Pre-loaded failure episodes (alternative to failure_file_path)
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.use_all_states = use_all_states
        self.model_name = model_name
        
        print(f" Training T_Value for: [{model_name}]")

        # Determine which keys to use
        if self.use_all_states:
            self.state_keys = None
            print("Using ALL available states")
        elif state_keys is not None:
            self.state_keys = state_keys
            print(f"Using states: {state_keys}")
        else:
            self.state_keys = ['cube']
            print(f"Using default state: {self.state_keys}")
        
        # Load success episodes (from file or directly)
        if success_episodes is not None:
            print(f"Using pre-loaded success episodes ({len(success_episodes)} episodes)")
        elif success_file_path is not None:
            print(f"Loading success data from {success_file_path}")
            success_episodes = torch.load(success_file_path, map_location=self.device)
        else:
            raise ValueError("Must provide either success_file_path or success_episodes")

        # Load failure episodes (from file or directly)
        if failure_episodes is not None:
            print(f"Using pre-loaded failure episodes ({len(failure_episodes)} episodes)")
        elif failure_file_path is not None:
            print(f"Loading failure data from {failure_file_path}")
            failure_episodes = torch.load(failure_file_path, map_location=self.device)
        else:
            raise ValueError("Must provide either failure_file_path or failure_episodes")

        # Extract relevant states from episodes
        self.success_data = self._extract_states_from_episodes(success_episodes)
        print(f"Loaded {len(self.success_data)} successful states")

        self.failure_data = self._extract_states_from_episodes(failure_episodes)
        print(f"Loaded {len(self.failure_data)} failed states")

        # Split data into train and validation using RANDOM sampling (not sequential)
        # This ensures validation data comes from all strategies, not just the last one
        num_success_validation = int(len(self.success_data) * validation_split)
        num_failure_validation = int(len(self.failure_data) * validation_split)

        # Generate random indices for validation split
        success_indices = torch.randperm(len(self.success_data))
        failure_indices = torch.randperm(len(self.failure_data))
        
        # Split using random indices
        valid_success_idx = success_indices[:num_success_validation]
        train_success_idx = success_indices[num_success_validation:]
        valid_failure_idx = failure_indices[:num_failure_validation]
        train_failure_idx = failure_indices[num_failure_validation:]
        
        self.valid_data_success = self.success_data[valid_success_idx].clone()
        self.success_data = self.success_data[train_success_idx].clone()
        self.valid_data_fail = self.failure_data[valid_failure_idx].clone()
        self.failure_data = self.failure_data[train_failure_idx].clone()
        
        print(f"✓ Validation split uses random sampling across all strategies")
                
        self.valid_data = torch.cat([self.valid_data_success, self.valid_data_fail], dim=0)
        self.valid_labels = torch.cat([
            torch.ones(len(self.valid_data_success), dtype=torch.float32, device=self.device),
            torch.zeros(len(self.valid_data_fail), dtype=torch.float32, device=self.device)
        ], dim=0)

        self.num_success_data = len(self.success_data)
        self.num_failure_data = len(self.failure_data)

        # Compute normalization stats from TRAINING data only (not validation)
        all_train_data = torch.cat([self.success_data, self.failure_data], dim=0)
        self.obs_mean = all_train_data.mean(dim=0, keepdim=True)
        self.obs_std = all_train_data.std(dim=0, keepdim=True) + 1e-6

        self.input_dim = self.success_data.shape[1]
        
        # Print dataset statistics
        train_success_rate = len(self.success_data) / (len(self.success_data) + len(self.failure_data))
        valid_success_rate = len(self.valid_data_success) / len(self.valid_data)
        print(f"Input dimension: {self.input_dim}")
        print(f"Train: {len(self.success_data)} success, {len(self.failure_data)} failure (success rate: {train_success_rate:.2%})")
        print(f"Valid: {len(self.valid_data_success)} success, {len(self.valid_data_fail)} failure (success rate: {valid_success_rate:.2%})\n")

    def _extract_states_from_episodes(self, episodes: List[List[Dict]]) -> torch.Tensor:
        """
        Extract relevant states from episode data.
        
        Args:
            episodes: List of episodes, where each episode is a list of state dicts
            
        Returns:
            Tensor of shape (num_states, state_dim) containing extracted features
        """        
        # Determine which keys to use from first state if use_all_states
        if self.use_all_states and self.state_keys is None:
            sample_state = episodes[0][0]
            self.state_keys = list(sample_state.keys())
            print(f"Auto-detected state keys: {self.state_keys}")
            
        all_sequences = []
        for episode in episodes:
            episode_components = []
            for timestep_state in episode:
                state_components = []
                for key in self.state_keys:
                    if key not in timestep_state:
                        raise ValueError(f"Key '{key}' not found in state. Available: {list(timestep_state.keys())}")
                    
                    state_tensor = timestep_state[key]
                    if state_tensor.dim() == 0:
                        state_tensor = state_tensor.unsqueeze(0)
                    elif state_tensor.dim() > 1:
                        state_tensor = state_tensor.flatten()
                    state_components.append(state_tensor.to(self.device))
                episode_components.append(torch.cat(state_components))
            all_sequences.append(torch.cat(episode_components))
        
        if len(all_sequences) == 0:
            raise ValueError(f"No states found. Check state_keys: {self.state_keys}")
        
        return torch.stack(all_sequences)

    def init_model(self, rollout: int, 
                    learning_rate: float = 0.001,
                    batch_size: int = 1024):
        """
        Initialize the terminal value function and training parameters.
        
        Args:
            rollout: Number of training iterations
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
        """
        
        self.t_value = GraspInsertTValue(input_dim=self.input_dim, output_dim=1).to(self.device)
        for param in self.t_value.parameters():
            param.requires_grad_(True)
    
        self.t_value_optimizer = optim.Adam(self.t_value.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        self.t_value_save_path = f"./intermediate_value_function/{self.model_name}/"
        os.makedirs(self.t_value_save_path, exist_ok=True)
        
        # Calculate class weights for imbalanced data
        total = self.num_success_data + self.num_failure_data
        
        # pos_weight in BCEWithLogitsLoss weights the POSITIVE class (label=1, i.e., success)
        pos_weight = torch.tensor([self.num_failure_data / self.num_success_data], device=self.device)
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        success_rate = self.num_success_data / total
        failure_rate = self.num_failure_data / total
        print(f"Data distribution - Success: {success_rate:.1%}, Failure: {failure_rate:.1%}")
        print(f"BCE pos_weight: {pos_weight.item():.3f}")

        
        self.batch_size = batch_size
        self.valid_batch_size = min(4096, len(self.valid_data))
        self.rollout = rollout

        self.success_buf = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=self.device)

        self.succ_rand_range = range(0, self.num_success_data)
        self.fail_rand_range = range(0, self.num_failure_data)

        self.t_value_obs_buf = torch.zeros((self.batch_size, self.input_dim), 
                                          dtype=torch.float32, device=self.device)
        self.valid_t_value_obs_buf = torch.zeros((self.valid_batch_size, self.input_dim), 
                                                 dtype=torch.float32, device=self.device)

        print(f"Model initialized | Save path: {self.t_value_save_path}\n")


    def train(self, save_freq: int = 10000, augmentation_noise: float = 0.00):
        """
        Train the T-Value function.
        
        Args:
            save_freq: How often to save checkpoints and print validation metrics
            augmentation_noise: Amount of random noise to add for data augmentation
        """
        best_accuracy = -1.0
        train_losses = []

        for iter in range(self.rollout):
            rand_float = torch_rand_float(-1, 1, (self.batch_size, self.input_dim), device=self.device)

            # Sample according to natural distribution
            total_data = self.num_success_data + self.num_failure_data
            success_ratio = self.num_success_data / total_data
            num_success_in_batch = int(self.batch_size * success_ratio)
            num_failure_in_batch = self.batch_size - num_success_in_batch
            
            succ_rand = random.sample(self.succ_rand_range, num_success_in_batch)
            fail_rand = random.sample(self.fail_rand_range, num_failure_in_batch)
            
            self.t_value_obs_buf[:num_success_in_batch] = self.success_data[succ_rand]
            self.t_value_obs_buf[num_success_in_batch:] = self.failure_data[fail_rand]
            
            # Set labels
            self.success_buf[:num_success_in_batch] = 1.0
            self.success_buf[num_success_in_batch:] = 0.0

            # Normalize and augment
            self.t_value_obs_buf = (self.t_value_obs_buf - self.obs_mean) / self.obs_std
            self.t_value_obs_buf += rand_float * augmentation_noise

            predict_success = self.t_value(self.t_value_obs_buf)
            loss = self.bce_logits_loss(predict_success, self.success_buf)
            
            self.t_value_optimizer.zero_grad()
            loss.backward()
            self.t_value_optimizer.step()
            
            train_losses.append(loss.item())

            # Validation and saving
            if iter % save_freq == 0 or iter == self.rollout - 1:
                # Calculate train accuracy on recent batches
                with torch.no_grad():
                    train_logits = self.t_value(self.t_value_obs_buf)
                    train_probs = torch.sigmoid(train_logits)
                    train_pred = (train_probs > 0.5).float()
                    train_acc = (train_pred == self.success_buf).float().mean().item()
                
                # Validation (NO augmentation noise!)
                valid_rand = random.sample(range(len(self.valid_data)), self.valid_batch_size)
                self.valid_t_value_obs_buf = (self.valid_data[valid_rand] - self.obs_mean) / self.obs_std

                valid_logits = self.t_value(self.valid_t_value_obs_buf).detach()
                true_success = self.valid_labels[valid_rand].unsqueeze(1)

                valid_loss = self.bce_logits_loss(valid_logits, true_success)
                
                valid_probs = torch.sigmoid(valid_logits)
                pred_success = (valid_probs > 0.5).float()

                # Calculate metrics
                TP = ((pred_success == 1) & (true_success == 1)).sum().item()
                FP = ((pred_success == 1) & (true_success == 0)).sum().item()
                FN = ((pred_success == 0) & (true_success == 1)).sum().item()
                TN = ((pred_success == 0) & (true_success == 0)).sum().item()

                accuracy = (TP + TN) / (TP + FP + FN + TN)
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                avg_train_loss = np.mean(train_losses[-save_freq:]) if train_losses else loss.item()

                print(f"Iter {iter:6d} | Train Acc: {train_acc:.4f} Loss: {avg_train_loss:.4f} | "
                      f"Val Acc: {accuracy:.4f} Loss: {valid_loss.item():.4f} | "
                      f"P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f}")

                torch.save({
                    'model_state_dict': self.t_value.state_dict(),
                    'obs_mean': self.obs_mean,
                    'obs_std': self.obs_std,
                    'model_name': self.model_name,
                    'state_keys': self.state_keys,
                    'input_dim': self.input_dim,
                    'iter': iter,
                    'accuracy': accuracy,
                }, f"{self.t_value_save_path}/{iter}.pt")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save({
                        'model_state_dict': self.t_value.state_dict(),
                        'obs_mean': self.obs_mean,
                        'obs_std': self.obs_std,
                        'model_name': self.model_name,
                        'state_keys': self.state_keys,
                        'input_dim': self.input_dim,
                        'iter': iter,
                        'accuracy': accuracy,
                    }, f"{self.t_value_save_path}/best.pt")

                if iter == self.rollout - 1:
                    print(f"\n{'='*60}")
                    print(f"FINAL EVALUATION")
                    print(f"{'='*60}")
                    print(f"Best Val Acc: {best_accuracy:.4f}")
                    print(f"Final Val Acc: {accuracy:.4f} | TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
                    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
                    print(f"{'='*60}\n")

    def test_model(self, test_data_path: str, 
                    image_dir: str, 
                    save_dir: str, 
                    threshold: float = 0.8,
                    checkpoint_iter: Optional[int] = None):
        """
        Test the trained model and visualize predictions on images.
        
        Args:
            test_data_path: Path to test data .pt file
            image_dir: Directory containing images
            save_dir: Directory to save visualized results
            threshold: Confidence threshold for coloring
            checkpoint_iter: Which checkpoint to load (defaults to final)
        """
        try:
            import cv2
        except ImportError:
            print("CV2 not found. Install opencv-python to run visualization.")
            return

        if checkpoint_iter is None:
            checkpoint_iter = self.rollout - 1
        
        model_path = os.path.join(self.t_value_save_path, f"{checkpoint_iter}.pt")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.obs_mean = checkpoint['obs_mean'].to(self.device)
        self.obs_std = checkpoint['obs_std'].to(self.device)

        self.t_value = GraspInsertTValue(input_dim=self.input_dim, output_dim=1).to(self.device)
        self.t_value.load_state_dict(checkpoint['model_state_dict'])
        self.t_value.to(self.device)
        self.t_value.eval()

        print(f"Loading test data from {test_data_path}")

        episodes = torch.load(test_data_path, map_location=self.device)
        all_states = self._extract_states_from_episodes(episodes)

        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            all_states = (all_states - self.obs_mean) / self.obs_std
            logits = self.t_value(all_states)
            success_probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        print(f"Processing {len(all_states)} states for visualization...")

        for i in range(len(all_states)):
            image_path = os.path.join(image_dir, f"episode_{i}_final.png")
            
            if not os.path.exists(image_path):
                print(f"Image not found for state {i} at {image_path}. Skipping.")
                continue 

            image = cv2.imread(image_path)
            prob = success_probs[i]
            text = f"{self.model_name} | P(Success): {prob:.4f}"
            
            if prob > threshold:
                color = (0, 255, 0)  # Green for high success
            elif prob < (1.0 - threshold):
                color = (0, 0, 255)  # Red for high failure
            else:
                color = (255, 255, 0)  # Yellow for uncertain

            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)            
            save_path = os.path.join(save_dir, f"{self.model_name}_{i:05d}_pred.png")
            cv2.imwrite(save_path, image)
        
        cv2.destroyAllWindows()
        print("Visualization complete!\n")

def load_and_combine_episodes(file_paths: List[str], device: str = "cuda:0", 
                              track_strategy: bool = False) -> List:
    """
    Load and combine multiple episode files.
    
    Args:
        file_paths: List of paths to episode files
        device: Device to load tensors to
        track_strategy: If True, add 'strategy_id' to each timestep for tracking
        
    Returns:
        Combined list of episodes
    """
    all_episodes = []
    for strategy_id, path in enumerate(file_paths):
        episodes = torch.load(path, map_location=device)
        
        # Optionally tag each timestep with its strategy ID for tracking
        if track_strategy:
            for episode in episodes:
                for timestep in episode:
                    timestep['strategy_id'] = torch.tensor(strategy_id, device=device)
        
        all_episodes.extend(episodes)
        print(f"  Strategy {strategy_id}: loaded {len(episodes)} episodes from {path.split('/')[-1]}")
    
    print(f"Combined {len(file_paths)} files into {len(all_episodes)} total episodes")
    return all_episodes

def load_tvalue_model(model_name: str, checkpoint_iter: Optional[int] = None) -> Tuple:
    """
    Load a trained T-Value model for a specific strategy.
    
    Args:
        model_name: Name of model to load
        strategy_name: Strategy name (e.g., "index_middle")
        checkpoint_iter: Which checkpoint to load (defaults to finding latest)
        
    Returns:
        Tuple of (model, obs_mean, obs_std, state_keys)
    """
    save_path = f"./intermediate_value_function/{model_name}/"
    
    if not os.path.exists(save_path):
        raise ValueError(f"No T-Value model found at {save_path}")
    
    if checkpoint_iter is None:
        checkpoints = [f for f in os.listdir(save_path) if f.endswith('.pt')]
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {save_path}")
        checkpoint_iter = max([int(f.split('.')[0]) for f in checkpoints])
    
    checkpoint_path = os.path.join(save_path, f"{checkpoint_iter}.pt")
    checkpoint = torch.load(checkpoint_path)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = GraspInsertTValue(
        input_dim=checkpoint['input_dim'],
        output_dim=1
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    obs_mean = checkpoint['obs_mean'].to(device)
    obs_std = checkpoint['obs_std'].to(device)
    state_keys = checkpoint.get('state_keys', None)
    
    print(f"Loaded T-Value model '{model_name}' from checkpoint {checkpoint_iter}")
    if state_keys:
        print(f"  State keys: {state_keys}")
    
    return model, obs_mean, obs_std, state_keys



def compute_reward(state: torch.Tensor, model, obs_mean: torch.Tensor, 
                  obs_std: torch.Tensor) -> float:
    """
    Compute T-Value reward for a given state.
    Use this as reward shaping during Task 1 training.
    
    Args:
        state: Current state tensor
        t_value_model: Loaded T-Value model
        obs_mean: Mean for normalization
        obs_std: Std for normalization
        
    Returns:
        Success probability (0-1)
    """
    with torch.no_grad():
        normalized = (state - obs_mean) / obs_std
        logit = model(normalized.unsqueeze(0))
        prob = torch.sigmoid(logit).item()
    return prob


if __name__ == "__main__":
    
    state_keys = ['cube', 'goal_site', 'xarm7_leap_right']

    validation_folder = "intermediate_states/fingers_new_urdf3_0_1_2_active_2_palm_False_seed_8"

    # Single strategy
    trainer = TValue_Trainer(
        success_file_path="relabeled_states/fingers_new_urdf3_0_1_2_active_2_palm_False_seed_8/xArm7-v1-push_fingers_new_urdf3_0_1_2_active_2_palm_False_seed_8/success_states.pt",
        failure_file_path="relabeled_states/fingers_new_urdf3_0_1_2_active_2_palm_False_seed_8/xArm7-v1-push_fingers_new_urdf3_0_1_2_active_2_palm_False_seed_8/failure_states.pt",
        model_name="push_index_thumb",
        state_keys=state_keys,
        use_all_states=False
    )
    trainer.init_model(rollout=100000, learning_rate=0.001, batch_size=1024)
    trainer.train(save_freq=5000)

    
    # # Unified model (combine all strategies)
    # strategies = [
    #     "relabeled_states/fingers_new_urdf1_2_0_3_active_2_palm_True_seed_8/xArm7-v1-push_fingers_new_urdf1_2_0_3_active_2_palm_True_seed_8",
    #     "relabeled_states/fingers_new_urdf2_0_1_3_active_1_palm_True_seed_8/xArm7-v1-push_fingers_new_urdf2_0_1_3_active_1_palm_True_seed_8",
    #     "relabeled_states/fingers_new_urdf3_0_1_2_active_2_palm_False_seed_8/xArm7-v1-push_fingers_new_urdf3_0_1_2_active_2_palm_False_seed_8",
    #     "relabeled_states/fingers_new_urdf3_2_0_1_active_2_palm_False_seed_8/xArm7-v1-push_fingers_new_urdf3_2_0_1_active_2_palm_False_seed_8",
    # ]
    # success_files = [f"{s}/success_states.pt" for s in strategies]
    # failure_files = [f"{s}/failure_states.pt" for s in strategies]
    
    # # Load and combine episodes directly
    # combined_success = load_and_combine_episodes(success_files)
    # combined_failure = load_and_combine_episodes(failure_files)
    
    # # Train unified model directly with combined episodes (no temp files needed!)
    # trainer = TValue_Trainer(
    #     success_episodes=combined_success,
    #     failure_episodes=combined_failure,
    #     model_name="push_unified",
    #     state_keys=state_keys,
    #     use_all_states=False
    # )
    # trainer.init_model(rollout=50000, learning_rate=0.001, batch_size=1024)
    # trainer.train(save_freq=5000)


    trainer.test_model(
        test_data_path=f"{validation_folder}/grasp_to_push_img.pt",
        image_dir=f"{validation_folder}/images",
        save_dir=f"{validation_folder}/success_predictions",
        threshold=0.8,
        checkpoint_iter=None 
    )