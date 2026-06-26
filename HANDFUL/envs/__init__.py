# Import all task environments to register them with Gym / ManiSkill
from envs.tasks import xarm7_allegron_env
from envs.tasks import xarm7_leap_pick_env
from envs.tasks import xarm7_leap_push_env
from envs.tasks import xarm7_open_drawer
from envs.tasks import xarm7_pick_all
from envs.tasks import xarm7_pick_randomized
from envs.tasks import xarm7_press_button
from envs.tasks import xarm7_twist
from envs.tasks import xarm7_two_pick

from envs.tasks.unified_environments import xarm7_open_drawer_unified
from envs.tasks.unified_environments import xarm7_press_button_unified
from envs.tasks.unified_environments import xarm7_push_unified
from envs.tasks.unified_environments import xarm7_twist_unified
from envs.tasks.unified_environments import xarm7_two_pick_unified

from envs.tasks.whole_hand_environments import xarm7_open_drawer_whole_hand
from envs.tasks.whole_hand_environments import xarm7_press_button_whole_hand
from envs.tasks.whole_hand_environments import xarm7_push_whole_hand
from envs.tasks.whole_hand_environments import xarm7_twist_whole_hand
from envs.tasks.whole_hand_environments import xarm7_two_pick_whole_hand
