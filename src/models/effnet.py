import torch
from efficientnet_pytorch import EfficientNet


class Model(torch.nn.Module):
    def __init__(self, n_classes=3):
        super(Model, self).__init__()
        self.n_classes = n_classes
        
        self.base_model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=self.n_classes)

    def forward(self, x):
        logits = self.base_model(x)

        return logits


if __name__ == '__main__':
    model = Model()
    
    print(model(torch.randn(3, 3, 100, 100)))
    # print(model)

