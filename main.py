import gymnasium as gym
import numpy as np

# Importing the envs folder triggers your __init__.py file, 
# which registers 'VisionAnnotator-v0' with Gymnasium.
import envs 

def llm_judge(action_log, final_iou):
    """
    Simulates calling an LLM API (like Llama 3 or GPT-4) to grade the agent.
    In the hackathon, replace this mock logic with your actual API request.
    """
    prompt = f"""
    You are an AI code reviewer grading a computer vision agent.
    The agent was tasked with adjusting a bounding box to achieve a high Intersection over Union (IoU).
    
    Final IoU achieved: {final_iou:.2f}
    
    Agent Action Log:
    {action_log}
    
    Grade the agent's strategy on a scale of 1 to 10. 
    Did it move erratically, or did it systematically zero in on the target?
    """
    
    # TODO: Send 'prompt' to your chosen LLM API here.
    # For now, we will mock a response:
    mock_score = 7.5
    mock_feedback = "The agent made progress but took a few erratic steps horizontally."
    
    return mock_score, mock_feedback


def main():
    print("Initializing VisionAnnotator-v0...")
    env = gym.make('VisionAnnotator-v0')
    
    # 1. Start the episode
    obs, info = env.reset()
    done = False
    truncated = False
    
    total_programmatic_reward = 0.0
    action_log = ""
    final_iou = 0.0
    
    print("\n--- Agent Execution Loop ---")
    
    # 2. The Game Loop
    while not (done or truncated):
        # We are using a random agent for testing.
        # Later, replace this with: action = trained_model.predict(obs)
        action = env.action_space.sample() 
        
        # Take the step
        obs, reward, done, truncated, info = env.step(action)
        
        total_programmatic_reward += reward
        final_iou = info.get("iou", 0.0)
        
        # Log the action for the LLM
        step_log = f"Step {env.unwrapped.current_step}: Shifted box by [dx:{action[0]:.1f}, dy:{action[1]:.1f}, dw:{action[2]:.1f}, dh:{action[3]:.1f}] -> New IoU: {final_iou:.2f}\n"
        action_log += step_log
        print(step_log.strip())
        
    print("\n--- Evaluation Phase ---")
    print(f"Total Dense Reward (Programmatic): {total_programmatic_reward:.2f}")
    print(f"Final Bounding Box IoU: {final_iou:.2f}")
    
    # 3. Trigger the LLM Judge
    print("\nConsulting LLM Judge...")
    llm_score, llm_feedback = llm_judge(action_log, final_iou)
    
    print(f"LLM Score: {llm_score}/10")
    print(f"LLM Feedback: {llm_feedback}")
    
    # 4. Calculate Final Hackathon Score
    # Weighting: 60% Programmatic Success, 40% LLM Evaluation of Strategy
    normalized_iou_score = final_iou * 10  # Scale 0-1 to 0-10
    final_combined_score = (normalized_iou_score * 0.6) + (llm_score * 0.4)
    
    print("\n========================================")
    print(f"FINAL SUBMISSION SCORE: {final_combined_score:.2f} / 10.00")
    print("========================================")

if __name__ == "__main__":
    main()