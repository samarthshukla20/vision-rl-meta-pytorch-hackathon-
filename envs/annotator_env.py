import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import cv2
import glob

from .utils import calculate_iou

class VisionAnnotatorEnv(gym.Env):
    """
    Mini-RL Environment for adjusting bounding boxes.
    Agent must maximize IoU against a hidden ground truth using YOLO datasets.
    """
    def __init__(self):
        super().__init__()
        
        # 1. THE ACTION AND OBSERVATION SPACES
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(4,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            'current_box': spaces.Box(low=0.0, high=224.0, shape=(4,), dtype=np.float32) 
        })
        
        self.IMAGE_SIZE = 224.0
        self.current_step = 0
        self.max_steps = 10 
        
        # 2. YOLO DATASET PATHS
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.train_images_dir = os.path.join(current_dir, '..', 'data', 'images', 'train')
        self.train_labels_dir = os.path.join(current_dir, '..', 'data', 'labels', 'train')
        
        self.image_paths = glob.glob(os.path.join(self.train_images_dir, '*.[jp][pn]g'))
        
        if len(self.image_paths) > 0:
            print(f"Successfully found {len(self.image_paths)} images in YOLO format.")
        else:
            print(f"WARNING: No images found at {self.train_images_dir}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # 3. LOAD RANDOM IMAGE AND YOLO LABEL
        if len(self.image_paths) > 0:
            img_path = self.np_random.choice(self.image_paths)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.current_image = cv2.resize(img, (int(self.IMAGE_SIZE), int(self.IMAGE_SIZE)))
            
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            label_path = os.path.join(self.train_labels_dir, f"{base_name}.txt")
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    line = f.readline().strip()
                    parts = line.split()
                    
                    if len(parts) >= 5:
                        x_center_norm, y_center_norm, w_norm, h_norm = map(float, parts[1:5])
                        
                        abs_w = w_norm * self.IMAGE_SIZE
                        abs_h = h_norm * self.IMAGE_SIZE
                        abs_x_center = x_center_norm * self.IMAGE_SIZE
                        abs_y_center = y_center_norm * self.IMAGE_SIZE
                        
                        x_min = abs_x_center - (abs_w / 2.0)
                        y_min = abs_y_center - (abs_h / 2.0)
                        
                        self.ground_truth_box = np.array([x_min, y_min, abs_w, abs_h], dtype=np.float32)
            else:
                self.ground_truth_box = np.array([50.0, 50.0, 100.0, 100.0], dtype=np.float32)
        else:
            self.current_image = np.full((int(self.IMAGE_SIZE), int(self.IMAGE_SIZE), 3), 128, dtype=np.uint8)
            self.ground_truth_box = np.array([87.0, 87.0, 50.0, 50.0], dtype=np.float32)

        # 4. GENERATE NOISY STARTING BOX
        noise = self.np_random.uniform(low=-5.0, high=5.0, size=(4,)).astype(np.float32)
        self.current_box = self.ground_truth_box + noise
        self._clip_box()

        # Fix Gymnasium float64 warnings
        self.current_box = self.current_box.astype(np.float32)

        observation = {
            'image': self.current_image.copy(),
            'current_box': self.current_box.copy()
        }
        return observation, {}

    def step(self, action):
        """Applies the agent's action and returns the IoU reward."""
        self.current_step += 1
        
        old_iou = calculate_iou(self.current_box, self.ground_truth_box)
        
        self.current_box += action
        self._clip_box()
        
        # Fix Gymnasium float64 warnings
        self.current_box = self.current_box.astype(np.float32)
        
        new_iou = calculate_iou(self.current_box, self.ground_truth_box)
        reward = (new_iou - old_iou) * 10.0 
        
        is_perfect = new_iou > 0.95
        done = bool(self.current_step >= self.max_steps or is_perfect)
        
        if is_perfect:
            reward += 10.0 
            
        observation = {
            'image': self.current_image.copy(),
            'current_box': self.current_box.copy()
        }
        
        return observation, reward, done, False, {"iou": new_iou}

    def _clip_box(self):
        """Helper function to prevent the box from leaving the image bounds."""
        self.current_box[0] = np.clip(self.current_box[0], 0, self.IMAGE_SIZE - 1)
        self.current_box[1] = np.clip(self.current_box[1], 0, self.IMAGE_SIZE - 1)
        self.current_box[2] = np.clip(self.current_box[2], 1.0, self.IMAGE_SIZE - self.current_box[0])
        self.current_box[3] = np.clip(self.current_box[3], 1.0, self.IMAGE_SIZE - self.current_box[1])