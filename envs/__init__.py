from gymnasium.envs.registration import register
from .annotator_env import VisionAnnotatorEnv

# Register the environment with Gymnasium's global registry
register(
    id='VisionAnnotator-v0',
    entry_point='envs.annotator_env:VisionAnnotatorEnv',
    max_episode_steps=10, # A built-in failsafe to prevent infinite loops
)