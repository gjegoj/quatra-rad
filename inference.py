import cv2
import torch
import argparse
from pathlib import Path
from src.dataset.transforms import val_transform


ROOT = Path(__file__).resolve().parents[0]
CLASSES = {0: 'bottle', 1: 'glass', 2: 'packet'}
JIT_PATH = ROOT / 'jit/ResNet18.jit'

TEST_IMG_PATH = ROOT / 'data/downloaded_data/5_45140167_2486686-a2d111fe-e2a6-4508-aa76-57a7c9ebfdc7.jpg'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jit_path', type=str,
                        default=JIT_PATH, help='.jit file path')
    
    parser.add_argument('--image_path', type=str,
                        default=TEST_IMG_PATH, help='image file path')

    args = parser.parse_args()
    return args


def inference(image_path, model):

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = val_transform(image=image)['image']

    with torch.no_grad():
        logits = model(image.unsqueeze(0))
        preds = torch.argmax(torch.softmax(logits, dim=1), axis=1)

    return CLASSES[int(preds.numpy())]


args = parse_args()
base_model = torch.jit.load(args.jit_path)

if __name__ == '__main__':

    args = parse_args()

    base_model = torch.jit.load(args.jit_path)

    pred = inference(str(args.image_path), base_model)
    print(pred)
