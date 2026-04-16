import gymnasium as gym
import cv2
import numpy as np

# Registers the environment
import envs 

def test_render():
    print("Loading environment and fetching a random image...")
    env = gym.make('VisionAnnotator-v0')
    
    # Reset the environment to load a random image and its YOLO label
    obs, info = env.reset()
    
    # 1. Grab the image
    # The environment outputs RGB, but OpenCV's display functions expect BGR
    img = obs['image'].copy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 2. Get the boxes
    agent_box = obs['current_box'] # The "noisy" starting box
    gt_box = env.unwrapped.ground_truth_box # The perfect YOLO box
    
    # 3. Draw the Agent's noisy starting box (RED)
    cv2.rectangle(img_bgr, 
                  (int(agent_box[0]), int(agent_box[1])), 
                  (int(agent_box[0] + agent_box[2]), int(agent_box[1] + agent_box[3])), 
                  (0, 0, 255), 2)
                  
    # 4. Draw the Ground Truth box (GREEN)
    cv2.rectangle(img_bgr, 
                  (int(gt_box[0]), int(gt_box[1])), 
                  (int(gt_box[0] + gt_box[2]), int(gt_box[1] + gt_box[3])), 
                  (0, 255, 0), 2)
                  
    # Display the result
    print("Press any key on the image window to close it.")
    cv2.imshow("Hackathon Env Test (Green=Truth, Red=Start)", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_render()