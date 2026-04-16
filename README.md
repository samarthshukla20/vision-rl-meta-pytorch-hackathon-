# 🎯 Vision-RL: Auto-Annotator Environment
**Submission for the Meta PyTorch OpenEnv Hackathon x Scaler School of Technology**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-orange)
![Stable Baselines 3](https://img.shields.io/badge/SB3-PPO-brightgreen)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)

## 📖 Overview
The **Vision-RL Auto-Annotator** is a custom Reinforcement Learning (RL) environment built on the `gymnasium` framework. The environment simulates a computer vision annotation task where an AI agent is challenged to dynamically correct "noisy" or inaccurate bounding boxes to perfectly encapsulate objects in an image.

This project fulfills all hackathon requirements by implementing:
1. **Defined Tasks:** Spatial reasoning and coordinate adjustments on real YOLO datasets.
2. **Programmatic Graders:** Frame-by-frame Intersection over Union (IoU) calculation for dense reward shaping.
3. **LLM Scoring:** An integrated evaluation pipeline to judge the agent's spatial reasoning strategy based on its action logs.

---

## ✨ Key Features
* **Custom OpenEnv Physics (`VisionAnnotator-v0`):** A robust Gym environment tailored for computer vision.
* **YOLO Dataset Ingestion:** Automatically parses YOLO normalized `.txt` labels and translates them to absolute pixel coordinates for the RL state space.
* **Multi-Modal State Space:** The agent observes both a `224x224` RGB image tensor and a 4D coordinate array (`x, y, w, h`).
* **Dense Reward Shaping:** Utilizes frame-by-frame IoU deltas. The agent is actively penalized for shifting the box away from the target and rewarded for closing the gap.
* **PPO Training Ready:** fully integrated with Stable Baselines 3 using a `MultiInputPolicy` (CNN + MLP fusion network).

---

## 📂 Repository Structure

```text
vision-annotator-rl/
├── data/
│   ├── images/train/       # Place your .jpg/.png images here
│   └── labels/train/       # Place your YOLO format .txt labels here
├── envs/
│   ├── __init__.py         # Gymnasium environment registration
│   ├── annotator_env.py    # The core physics and reward engine
│   └── utils.py            # IoU mathematical calculations
├── main.py                 # Evaluation loop and LLM Judge API integration
├── test_visual.py          # OpenCV bounding box visual verification
├── train.py                # Stable Baselines 3 PPO training script
└── requirements.txt        # Project dependencies
```

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/vision-annotator-rl.git
   cd vision-annotator-rl
   ```

2. **Install dependencies:**
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install gymnasium opencv-python pandas "stable-baselines3[extra]"
   ```

3. **Add your Dataset:**
   Place your standard YOLO-formatted dataset inside the `data/` folder as shown in the repository structure.

---

## 🎮 Usage Guide

### 1. Visual Verification (Sanity Check)
Before training, verify that your YOLO coordinates are being perfectly translated to absolute pixels.
```bash
python test_visual.py
```
*Expected Output: An OpenCV window displaying a random dataset image with a **Green Box** (Ground Truth) and a **Red Box** (The noisy starting state for the agent).*

### 2. The Hackathon Evaluation Loop
Run the complete environment loop. This script drops an agent into the environment, evaluates its spatial adjustments programmatically (IoU), and passes the action log to an LLM for strategic scoring.
```bash
python main.py
```

### 3. Train the PPO Agent
Train a real AI to solve the environment using Proximal Policy Optimization (PPO).
```bash
python train.py
```
Monitor the agent's learning progress in real-time by opening a second terminal and running:
```bash
tensorboard --logdir ./ppo_vision_tensorboard/
```

---

## 🧠 Environment Mechanics

* **Action Space:** `Box(-5.0, 5.0, (4,), float32)`
  * The agent outputs continuous adjustments `[dx, dy, dw, dh]` limited to 5 pixels per step to ensure stable learning.
* **Observation Space:** `Dict(image, current_box)`
  * `image`: `224x224x3` RGB tensor.
  * `current_box`: `[x_min, y_min, width, height]`.
* **Programmatic Reward:** * $+10.0 	imes \Delta	ext{IoU}$ (Dense shaping based on improvement).
  * $+10.0$ flat bonus for achieving $> 0.95$ IoU (Early completion).

---

## 🔮 Future Development
* **Integration of Llama 3 API:** Replacing the mocked LLM judge in `main.py` with Meta's Llama 3 API for live reasoning evaluation.
* **Hyperparameter Tuning:** Optimizing PPO's learning rate and batch size for faster convergence on complex visual datasets.

---
*Built for the Meta PyTorch OpenEnv Hackathon - April 2026*
