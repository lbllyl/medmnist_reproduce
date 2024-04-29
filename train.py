import torch
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm

from medmnist import PathMNIST, ChestMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, BreastMNIST, DermaMNIST
from torchvision import models, datasets, transforms
import time

def load_model(model_name, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # 修改全连接层以适应你的特定类别数
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("configs/config.json", "r") as f:
        config = json.load(f)

    datasets_name = config["datasets_name"]

    # train_loader, val_loader = get_data_loaders(config["data_dir"], config["batch_size"], config["num_workers"])

    # torchvision加载mnist
    if datasets_name == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
    # medmnist加载数据集
    elif datasets_name == "pathmnist":
        train_loader = torch.utils.data.DataLoader(
            PathMNIST(root='data', split='train', download=True, transform=transforms.Compose([
                transforms.RandomRotation(degrees=10),  # 随机旋转±10度
                transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            PathMNIST(root='data', split='val', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            PathMNIST(root='data', split='test', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
    elif datasets_name == "chestmnist":
        train_loader = torch.utils.data.DataLoader(
            ChestMNIST(root='data', split='train', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            ChestMNIST(root='data', split='val', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            ChestMNIST(root='data', split='test', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
    elif datasets_name == "octmnist":
        train_loader = torch.utils.data.DataLoader(
            OCTMNIST(root='data', split='train', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            OCTMNIST(root='data', split='val', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            OCTMNIST(root='data', split='test', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
    elif datasets_name == "pneumoniamnist":
        train_loader = torch.utils.data.DataLoader(
            PneumoniaMNIST(root='data', split='train', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            PneumoniaMNIST(root='data', split='val', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            PneumoniaMNIST(root='data', split='test', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
    elif datasets_name == "retinamnist":
        train_loader = torch.utils.data.DataLoader(
            RetinaMNIST(root='data', split='train', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            RetinaMNIST(root='data', split='val', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            RetinaMNIST(root='data', split='test', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
    elif datasets_name == "breastmnist":
        train_loader = torch.utils.data.DataLoader(
            BreastMNIST(root='data', split='train', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            BreastMNIST(root='data', split='val', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            BreastMNIST(root='data', split='test', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
    elif datasets_name == "dermamnist":
        train_loader = torch.utils.data.DataLoader(
            DermaMNIST(root='data', split='train', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            DermaMNIST(root='data', split='val', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            DermaMNIST(root='data', split='test', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=config["batch_size"], shuffle=True
        )
    else:
        raise ValueError(f"Unknown datasets name: {datasets_name}")

    # 加载模型
    model_name = "resnet18"  # 或 "resnet50"
    model = load_model(model_name, config["num_classes"]).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 50
    best_test_acc = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            target = target.squeeze()
            target = target.to(device).long()  # 确保目标张量是长整型
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

        # 验证循环
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        print(f'Validation Accuracy: {100. * correct / len(val_loader.dataset)}%')

        # 进行测试
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
        
        print(f'Test Accuracy: {100. * test_correct / len(test_loader.dataset)}%')

        if (100. * test_correct / len(test_loader.dataset)) > best_test_acc:
            best_test_acc = (100. * test_correct / len(test_loader.dataset))
            torch.save(model.state_dict(), f"{model_name}_best_model.pth")
            print("Model saved!")


        

    