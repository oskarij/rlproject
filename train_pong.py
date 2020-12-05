import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong

from agent import Agent

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--housekeeping", action="store_true", help="Plot, player and ball positions and velocities at the end of each episode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

def plot_reward_history(reward_history, episode_interval):
    plt.plot(reward_history)
    plt.xticks(np.arange(len(reward_history)), [f'{episode_interval * (i + 1)}' for i in range(len(reward_history))])
    plt.title(f'Average rewards for sequences of {episode_interval} episodes')
    plt.xlabel('Training episodes')
    plt.ylabel('Average rewards')
    plt.savefig('training_rewards.png')

# Number of episodes/games to play
EPISODES = 20000
EPISODE_INTERVAL = 100
TARGET_UPDATE_INTERVAL = 100
SAVE_INTERVAL = 1000
stack_frames =  4 # 0, 2,  4
observations_start_idx = 0 if stack_frames == 4 else 2

# Define the player
player_id = 1
player = Agent(training=True, stack_frames=stack_frames, downscale=True, priority_memory=True)

# Housekeeping
states = []
win1 = 0
reward_history = []
episode_rewards = 0

for i in range(1, EPISODES + 1):
    done = False
    observations = [None, None, None, None, None] # store 5 latest observations to update the agent
    frames = 0
    player.reset()
    while not done:
        action1 = player.get_action(observations[-1])
        observations[0] = observations[1]
        observations[1] = observations[2]
        observations[2] = observations[3]
        observations[3] = observations[4]
        observations[4], rew, done, info = env.step(action1)
        frames += 1
        if observations[observations_start_idx] is not None:
            player.update_memory(observations[observations_start_idx:], action1, rew, done)
        if frames % TARGET_UPDATE_INTERVAL == 0:
            player.update_target_network()

        player.update_dqn() # DQN update
        #player.update_ddqn() # Double DQN update
        episode_rewards += rew
        if args.housekeeping:
            states.append(observations[-1])
        # Count the wins
        if rew == 10:
            win1 += 1
        if not args.headless:
            env.render()
        if done:
            player.memory.update_beta()
            observation= env.reset()
            plt.close()  # Hides game window
            if args.housekeeping:
                plt.plot(states)
                plt.legend(["Player", "Opponent", "Ball X", "Ball Y", "Ball vx", "Ball vy"])
                plt.show()
                states.clear()
            print("episode {} over. Broken WR: {:.3f}, epsilon: {:.2f}".format(i, win1/(i+1), player.epsilon))
            if i % 5 == 4:
                env.switch_sides()
    if i % EPISODE_INTERVAL == 0:
        avg_rewards = episode_rewards / EPISODE_INTERVAL
        reward_history.append(avg_rewards)
        print(f'Average rewards for last {EPISODE_INTERVAL} episodes: {avg_rewards}')
        episode_rewards = 0
    if i % SAVE_INTERVAL == 0:
        player.save()
        plot_reward_history(reward_history, EPISODE_INTERVAL)
        if i == 12000:
            player.set_lr(1e-5)
        elif i == 8000:
            player.set_lr(2e-5)
        elif i == 5000:
            player.set_lr(4e-5)
player.save()
plot_reward_history(reward_history, EPISODE_INTERVAL)
