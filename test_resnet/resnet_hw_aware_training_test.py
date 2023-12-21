import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import numpy as np
import os
from typing import List
from matplotlib import rcParams
import matplotlib.pyplot as plt

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.presets.utils import IOParameters
from aihwkit.simulator.presets import StandardHWATrainingPreset
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import (
    WeightModifierType,
    BoundManagementType,
    WeightClipType,
    NoiseManagementType,
    WeightRemapType,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def _weights_init(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(self, block, num_blocks, n_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = torch.nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = torch.nn.Linear(64, n_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet32(n_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], n_classes=n_classes)


class TorchCutout(object):
    def __init__(self, length, fill=(0.0, 0.0, 0.0)):
        self.length = length
        self.fill = torch.tensor(fill).reshape(shape=(3, 1, 1))

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        img[:, y1:y2, x1:x2] = self.fill
        return img
    

def load_cifar10(batch_size, path):
    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
            TorchCutout(length=8),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    return trainloader, testloader


def load_cifar10_ffcv(batch_size, path):
    # - FFCV specific imports
    from ffcv.fields import IntField, RGBImageField
    from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.pipeline.operation import Operation
    from ffcv.transforms import (
        RandomHorizontalFlip,
        Cutout,
        RandomTranslate,
        Convert,
        ToDevice,
        ToTensor,
        ToTorchImage,
    )
    from ffcv.transforms.common import Squeeze
    from ffcv.writer import DatasetWriter

    datasets = {
        "train": torchvision.datasets.CIFAR10(path, train=True, download=True),
        "test": torchvision.datasets.CIFAR10(path, train=False, download=True),
    }

    for name, ds in datasets.items():
        writer = DatasetWriter(
            os.path.join(path, f"cifar_{name}.beton"),
            {"image": RGBImageField(), "label": IntField()},
        )
        writer.from_indexed_dataset(ds)

    # Note that statistics are wrt to uin8 range, [0,255].
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]

    loaders = {}
    for name in ["train", "test"]:
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice(device),
            Squeeze(),
        ]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == "train":
            image_pipeline.extend(
                [
                    RandomTranslate(padding=4),
                    RandomHorizontalFlip(),
                    Cutout(
                        8, tuple(map(int, CIFAR_MEAN))
                    ),  # - Note Cutout is done before normalization.
                ]
            )
        image_pipeline.extend(
            [
                ToTensor(),
                ToDevice(device, non_blocking=True),
                ToTorchImage(),
                Convert(torch.float32),
                torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

        # - Create loaders
        loaders[name] = Loader(
            os.path.join(path, f"cifar_{name}.beton"),
            batch_size=batch_size,
            num_workers=4,
            order=OrderOption.RANDOM,
            drop_last=(name == "train"),
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )

    return loaders["train"], loaders["test"]


def gen_rpu_config():
    rpu_config = InferenceRPUConfig()
    rpu_config.modifier.std_dev = 0.06
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL

    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = False
    rpu_config.mapping.out_scaling_columnwise = False
    rpu_config.remap.type = WeightRemapType.LAYERWISE_SYMMETRIC

    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
    rpu_config.clip.sigma = 2.0

    rpu_config.forward = IOParameters()
    rpu_config.forward.is_perfect = False
    rpu_config.forward.out_noise = 0.0
    rpu_config.forward.inp_bound = 1.0
    rpu_config.forward.inp_res = 1 / (2**8 - 2)
    rpu_config.forward.out_bound = 12
    rpu_config.forward.out_res = 1 / (2**8 - 2)
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.noise_management = NoiseManagementType.NONE

    rpu_config.pre_post.input_range.enable = True
    rpu_config.pre_post.input_range.decay = 0.01
    rpu_config.pre_post.input_range.init_from_data = 50
    rpu_config.pre_post.input_range.init_std_alpha = 3.0
    rpu_config.pre_post.input_range.input_min_percentage = 0.995
    rpu_config.pre_post.input_range.manage_output_clipping = False

    rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config


def train_step(model, optimizer, criterion, trainloader):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss / total, 100.0 * correct / total


def test_step(model, criterion, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(f"Test loss {test_loss/total:.4f} test acc. {100.*correct/total:.2f}%")
    return 100.0 * correct / total


def main():
    # - Set seeds
    torch.manual_seed(0)
    np.random.seed(0)
    # - Get the dataloader
    batch_size = 128
    trainloader, testloader = load_cifar10(
        batch_size=batch_size, path=os.path.expanduser("~/Data/")
    )

    # - hyperparameters
    lr = 0.05
    epochs = 100
    epochs_finetuning = 100

    # - Define model, criterion, optimizer and scheduler.
    model = resnet32()
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # - Pre-training of the network
    print("Pre-training of the network")
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, optimizer, criterion, trainloader)
        if epoch % 5 == 0:
            print(f"Pretrained test acc. {test_step(model, criterion, testloader)}%")
        scheduler.step()
    
    # - Fine-tuning
    print("Fine-tuning")
    analog_model = convert_to_analog(model, gen_rpu_config())
    optimizer = AnalogSGD(
        analog_model.parameters(), lr=lr / 10.0, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    test_accs = torch.empty(epochs_finetuning)
    for epoch in range(epochs_finetuning):
        train_loss, train_acc = train_step(analog_model, optimizer, criterion, trainloader)
        test_accs[epoch] = test_step(analog_model, criterion, testloader)
        print(f"Finetuned test acc. {test_accs[epoch]}%")
        scheduler.step()

    plt.title("Finetuning test accuracy")
    plt.plot(range(epochs_finetuning), test_accs, marker="d", linestyle="--", color="b")
    plt.ylabel("Test acc. (%)")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()
    plt.savefig("finetuning.png")


    # - Drift
    converted_model = convert_to_analog(model, StandardHWATrainingPreset())
    # - For programming the model, we need to put it into eval() mode
    converted_model = converted_model.eval()
    analog_model = analog_model.eval()
    # - We repeat each measurement 5 times
    n_rep = 5
    t_inferences = [60., 360., 8640., 25920., 31104.]
    drifted_test_accs = torch.zeros(size=(len(t_inferences),n_rep))
    drifted_test_accs_baseline = torch.zeros(size=(len(t_inferences),n_rep))
    for i,t in enumerate(t_inferences):
        for j in range(n_rep):
            converted_model.drift_analog_weights(t)
            drifted_test_accs_baseline[i,j] = test_step(converted_model, criterion, testloader)
            analog_model.drift_analog_weights(t)
            drifted_test_accs[i,j] = test_step(analog_model, criterion, testloader)
    plt.errorbar(t_inferences, drifted_test_accs.mean(dim=1).numpy(), drifted_test_accs.std(dim=1).numpy(), capsize=3, label="HW-aware")
    plt.errorbar(t_inferences, drifted_test_accs_baseline.mean(dim=1).numpy(), drifted_test_accs_baseline.std(dim=1).numpy(), capsize=3, label="Baseline")
    plt.xlabel("Time (s)")
    plt.xscale("log")
    plt.ylabel("Test acc. (%)")
    plt.legend()
    plt.show()
    plt.savefig("drift.png")


if __name__ == "__main__":
    main()
    