import torch

intermediate_states = torch.load("relabeled_states/xArm7-v1-pick-randomized__train__8__1766273487/xArm7-v1-two-pick__train__8__1766290217/failure_states.pt")

print(intermediate_states[0]) 