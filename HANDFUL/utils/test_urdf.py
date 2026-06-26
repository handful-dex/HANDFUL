import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agents.xarm7_allegro import XArm7Allegro# imports your robot and registers it
from agents.xarm7_leap import XArm7Leap  # imports the leap hand robot and registers it
# imports the demo_robot example script and lets you test your new robot
import mani_skill.examples.demo_robot as demo_robot_script
demo_robot_script.main()