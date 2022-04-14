import torch
import argparse
import numpy as np
from pathlib import Path
from src.dataset.transforms import val_transforms
from src.dataset.dataset import JapanItemsDataset 
from src.utils.plot import plot_roc_curve, plot_clf_report, plot_errors


ROOT = Path(__file__).resolve().parents[0]
CLASSES = {0: 'bottle', 1: 'glass', 2: 'packet'}
JIT_PATH = ROOT / 'jit/EffNetB3.jit'
# JIT_PATH = ROOT / 'jit/ResNet18.jit'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jit_path', type=str,
                        default=JIT_PATH, help='.jit file path')

    args = parser.parse_args()
    return args

args = parse_args()

dataset = JapanItemsDataset(is_train=False, transform=val_transforms())

base_model = torch.jit.load(args.jit_path)
base_model.eval()

targets, predicts  = [], []

preds_probs, one_hot_targets = [], []

errors = []
errors_targets, errors_preds = [], []

with torch.no_grad():
    for image, target in dataset:
        logits = base_model(image.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, axis=1)[0]

        targets.append(int(target))
        predicts.append(int(preds))

        one_hot_target = [0] * len(CLASSES)
        one_hot_target[int(target)] = 1

        one_hot_targets.append(one_hot_target)
        preds_probs.append(probs.tolist()[0])

        if int(target) != int(preds):
            errors.append(image.numpy())
            errors_targets.append(int(target))
            errors_preds.append(int(preds))


plot_errors(
            np.array(errors), 
            np.array(errors_targets), 
            np.array(errors_preds),
            save_path='docs/prediction_errors.jpg'
            )

plot_roc_curve(
                np.array(one_hot_targets), 
                np.array(preds_probs), 
                classes=CLASSES,
                save_path='docs/roc.jpg'
                )

plot_clf_report(
                targets, 
                predicts, 
                classes=CLASSES,
                save_path='docs/clf_report.jpg'
                )

