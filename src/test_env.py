import json
import ipdb as pdb
from src.envs.narrative_base_env import NarrativeEnv
from src.policies import random_policy, greedy_policy
from src.utils.helper import get_logger

logger = get_logger()


def run_episode(env, policy, **policy_kwargs):
    state = env.reset()
    done = False
    rewards = []
    episode_reward = 0

    while not done:
        action = policy(state, **policy_kwargs)
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        rewards.append(reward)

    return episode_reward


def run_episodes(env, policy, n_episodes, **policy_kwargs):
    avg_episode_reward = 0
    for _ in range(n_episodes):
        avg_episode_reward += run_episode(env, policy, **policy_kwargs)
    avg_episode_reward = avg_episode_reward / n_episodes
    logger.info(f"Average Episode Net Reward: {avg_episode_reward}")
    return avg_episode_reward


def main():

    MAX_SENTS = 1000
    N_EPISODES = 10

    # Load data
    with open("data/caanrc_tweets.json", "r") as f:
        all_tweets = json.load(f)
    all_tweets = list(all_tweets["full_text"].values())
    # pdb.set_trace()
    # Create environment
    env = NarrativeEnv(sentences=all_tweets[:MAX_SENTS])

    # Run Random Policy over environment
    logger.info("Running Random Policy on Narrative Environment")
    run_episodes(env, random_policy, n_episodes=N_EPISODES)
    # Run Greedy Policy with BLEU over environment
    logger.info("Running BLEU based Greedy Policy on Narrative Environment")
    run_episodes(env, greedy_policy, n_episodes=N_EPISODES, sim_type="bleu")

    # Run Greedy Policy with Embedding Similarity over environment
    logger.info(
        "Running Embedding-Similarity based Greedy Policy on Narrative Environment"
    )
    run_episodes(env, greedy_policy, n_episodes=N_EPISODES, sim_type="embedding_sim")


if __name__ == "__main__":
    main()
