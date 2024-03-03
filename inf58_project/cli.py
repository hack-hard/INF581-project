"""CLI interface for inf58_project project.

Be creative! do whatever you want!

- Install click or typer and create a CLI app
- Use builtin argparse
- Start a web application
- Import things from your .base module
"""
import gymnasium
import numpy
import time

def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m inf58_project` and `$ inf58_project `.

    This is your program's entry point.

    You can change this function to do whatever you want.
    Examples:
        * Run a test suite
        * Run a server
        * Do some other stuff
        * Run a command line application (Click, Typer, ArgParse)
        * List all available tasks
        * Run an application (Flask, FastAPI, Django, etc.)
    """
    env = gymnasium.make(
        id="ALE/Pacman-v5",
        full_action_space=False, # action space is Discrete(5) for NOOP, UP, RIGHT, LEFT, DOWN 
        render_mode="human",
        obs_type="ram", # observation_space=Box(0, 255, (128,), np.uint8)
        mode=0, # values in [0,...,7]
        difficulty=0 # values in [0,1]
    )
    
    for game in range(1): 
        obs = env.reset()
        done = False
        while not done: 
            env.render()
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
            time.sleep(1./30)
            print(reward)
    
    env.close()
    print("End of execution")

if __name__ == "__main__":  # pragma: no cover
    main()