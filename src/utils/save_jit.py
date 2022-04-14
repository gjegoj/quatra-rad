import os
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT)) if str(ROOT) not in sys.path else None

from src.models import resnet18, effnet

JIT_PATH = ROOT / 'jit'
JIT_NAME = 'EffNetB3.jit'
WEIGHTS = ROOT / 'runs/train/exp_7/weights/epoch=103.ckpt'

# Load model
# base_model = resnet18.Model()
base_model = effnet.Model(set_swith=False)

# Load weights
print(f"Loading from checkpoint: {WEIGHTS}")
model_weights = torch.load(WEIGHTS, map_location='cpu')['state_dict']
model_weights = {key[6:] :model_weights[key] for key in model_weights.keys()}
base_model.load_state_dict(model_weights, strict=False)
base_model.eval()
print('Weights were loaded!')

# Save jit
input = torch.rand((1, 3, 224, 224))
with torch.jit.optimized_execution(True):
    optimized_jit_model = torch.jit.trace(base_model, input)

torch.jit.save(optimized_jit_model, os.path.join(JIT_PATH, JIT_NAME))
print('Jit model created!')
