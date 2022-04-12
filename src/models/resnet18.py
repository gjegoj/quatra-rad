import torch, torchvision


class Model(torch.nn.Module):
    def __init__(self, n_classes=3):
        super(Model, self).__init__()
        self.n_classes = n_classes
        
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features, out_features=self.n_classes, bias=True)

    def forward(self, x):
        logits = self.base_model(x)
        
        return logits


if __name__ == '__main__':
    model = Model()

    print(model(torch.randn(3, 3, 100, 100)))
    # print(model)

