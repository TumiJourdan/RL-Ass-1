import torch
from stable_baselines3 import A2C
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
import wandb
from env_GNN import Gym2OpEnv  # Import your environment and custom extractor

def main():
    # Initialize Weights & Biases for tracking evaluation metrics
    run = wandb.init(
        project="Grid20p_Eval",
        name="evaluation",
        job_type="evaluation"
    )

    # Load the saved model
    model = A2C.load("A2C_GNN_NORM", device='cuda' if torch.cuda.is_available() else 'cpu')

    # Set up the environment
    env = Gym2OpEnv()
    env = Monitor(env)  # Monitor to track metrics

    total_episodes = 10  # Number of episodes for evaluation
    n_steps = 1000  # Max steps per episode

    for episode in range(total_episodes):
        obs = env.reset()
        total_reward = 0
        print(obs[0])
        first = True
        for step in range(n_steps):
            if (first):
                obs = obs[0]
                first = False
            action, _ = model.predict(obs, deterministic=True)  # Use deterministic actions
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if done or truncated:
                print(f"Episode {episode + 1} finished after {step + 1} steps")
                break

        # Log results per episode
        wandb.log({"episode": episode + 1, "total_reward": total_reward})

    print("Evaluation Completed!")
    run.finish()

if __name__ == "__main__":
    main()