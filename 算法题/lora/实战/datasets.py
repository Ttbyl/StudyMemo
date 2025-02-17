# 数据集选用CIFAR-100
import torchvision
import torchvision.transforms as transforms
from PIL import Image
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True, transform=transform, target_transform=None, download=True):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.data)