import random
from os import path as osp

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from torchvision import models

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

COLOR_JITTER_FACTOR = 0.10
DROPOUT = 0.40


def load_model(opt, model):
    if opt.model.path:
        pth = opt.model.path
    else:
        pth = f"models/{opt.model.arch}.pth.tar"

    if not osp.isfile(pth):
        raise ImportError(f"=> model checkpoint not found at '{pth}'")

    if not opt.compute.use_cuda:
        checkpoint = torch.load(pth, map_location=torch.device("cpu"))
        model = nn.DataParallel(model)
    elif opt.gpu is None:
        checkpoint = torch.load(pth)
    else:
        # Map model to be loaded to specified single gpu.
        loc = f"cuda:{opt.gpu}"
        checkpoint = torch.load(pth, map_location=loc)

    model.load_state_dict(checkpoint["state_dict"])
    print(f"=> loaded model '{pth}'")
    return model


def create_model(opt):
    if opt.model.arch.startswith("custom"):
        model = CustomNet(opt)
    elif opt.model.arch.startswith("resnet"):
        model = ResNet(opt)
    elif opt.model.arch == "random":
        model = Random(opt)
    else:
        raise NotImplementedError
    return model


class ResNet(nn.Module):
    def __init__(self, opt):
        super(ResNet, self).__init__()
        self.ensembling = False
        self.output_constraints = opt.model.output_constraints
        if opt.exp.task == "classification":
            self.n_classes = 5
            self.scale = torch.nn.Identity()
        elif opt.exp.task == "regression":
            self.n_classes = 37
            if self.output_constraints:
                self.scale = rescale_anwsers
            else:
                self.scale = nn.Sigmoid(dim=1)

        if opt.dataset.name == "imagenet":
            self.n_classes = 1000

        if opt.exp.test:
            self.scale = nn.Softmax(dim=1)
            self.ensembling = opt.ensembling.enable
            self.n_estimators = opt.ensembling.n_estimators
            opt.model.pretrained = False

        if opt.model.pretrained:
            print(f"=> using pre-trained model '{opt.model.arch}'")
            resnet = models.__dict__[opt.model.arch](pretrained=True)
        else:
            print(f"=> creating model '{opt.model.arch}'")
            resnet = models.__dict__[opt.model.arch]()

        num_ft = resnet.fc.in_features
        resnet.fc = torch.nn.Identity()
        m = [nn.Linear(num_ft, self.n_classes, bias=True)]  # , nn.ReLU(True)]
        # m = [
        #     nn.Linear(num_ft, 256, bias=True),
        #     nn.ReLU(),
        #     nn.Dropout(DROPOUT),
        #     nn.Linear(256, self.n_classes),
        # ]

        self.conv = resnet
        self.dense = nn.ModuleList(m)

        if opt.model.pretrained and opt.model.freeze:
            for param in self.conv.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ensembling:
            im_tf = EnsemblingTransforms()
            x_n = x.unsqueeze(0).repeat(self.n_estimators, 1, 1, 1, 1)
            x_out = torch.zeros((self.n_estimators, x.shape[0], self.n_classes))
            for i, x_ in enumerate(x_n):
                x_ = im_tf(x_)
                x_ = self._fwd(x_)
                x_out[i] = x_
            x = x_out.mean(axis=0)
        else:
            x = self._fwd(x)

        x = self.scale(x)
        return x

    def _fwd(self, x):
        x = self.conv(x)
        for m in self.dense:
            x = m(x)
        x = x.view(-1, self.n_classes)
        return x


class CustomNet(nn.Module):
    def __init__(self, opt):
        super(CustomNet, self).__init__()
        self.ensembling = False
        self.output_constraints = opt.model.output_constraints
        if opt.exp.task == "classification":
            self.n_classes = 5
            # self.scale = nn.Softmax(dim=1)
            self.scale = torch.nn.Identity()
        elif opt.exp.task == "regression":
            self.n_classes = 37
            if self.output_constraints:
                self.scale = rescale_anwsers
            else:
                self.scale = nn.Sigmoid(dim=1)

        if opt.dataset.name == "imagenet":
            self.n_classes = 1000

        if opt.exp.test:
            self.ensembling = opt.ensembling.enable
            self.n_estimators = opt.ensembling.n_estimators

        p = DROPOUT
        n = 0
        if len(opt.model.arch) > len("custom"):
            n = int(opt.model.arch[len("custom") :])

        m = []
        m.extend(
            [
                nn.BatchNorm2d(3),
                nn.Conv2d(3, 16, 5, padding=2, stride=2, bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 32, 3, padding=1, stride=2, bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 32, 3, padding=1, stride=1, bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                nn.Dropout(p),
                nn.Conv2d(64, 128, 3, padding=1, stride=2, bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(128),
                nn.Dropout(p),
                nn.Conv2d(128, 128, 3, padding=1, stride=2, bias=False),
                nn.ReLU(True),
            ]
        )
        for _ in range(n):
            m.extend(
                [
                    nn.BatchNorm2d(128),
                    nn.Dropout(p),
                    nn.Conv2d(128, 128, 3, padding=1, stride=1, bias=False),
                    nn.ReLU(True),
                ]
            )
        m.extend(
            [
                nn.BatchNorm2d(128),
                nn.Dropout(p),
                nn.Conv2d(128, 256, 3, padding=1, stride=2, bias=False),
                nn.ReLU(True),
            ]
        )
        self.conv = nn.ModuleList(m)
        m = [
            nn.BatchNorm2d(256 * 4 * 4),
            nn.Conv2d(256 * 4 * 4, self.n_classes, 1, padding=0, stride=1, bias=False),
            nn.ReLU(True),
        ]
        self.dense = nn.ModuleList(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ensembling:
            im_tf = EnsemblingTransforms()
            x_n = x.unsqueeze(0).repeat(self.n_estimators, 1, 1, 1, 1)
            x_out = torch.zeros((self.n_estimators, x.shape[0], self.n_classes))
            for i, x_ in enumerate(x_n):
                x_ = im_tf(x_)
                x_ = self._fwd(x_)
                x_out[i] = x_
            x = x_out.mean(axis=0)
        else:
            x = self._fwd(x)

        x = self.scale(x)
        return x

    def _fwd(self, x):
        for m in self.conv:
            x = m(x)
        # x = x.contiguous()
        x = x.view(-1, 256 * 4 * 4, 1, 1)

        for m in self.dense:
            x = m(x)
        x = x.view(-1, self.n_classes)

        return x


def rescale_anwsers(x):
    # raw model outputs ReLu-ised to questions
    q1 = x[:, :3]
    q2 = x[:, 3:5]
    q3 = x[:, 5:7]
    q4 = x[:, 7:9]
    q5 = x[:, 9:13]
    q6 = x[:, 13:15]
    q7 = x[:, 15:18]
    q8 = x[:, 18:25]
    q9 = x[:, 25:28]
    q10 = x[:, 28:31]
    q11 = x[:, 31:]

    # descision-tree-prior encoding
    eps = 1e-12
    q1_scaler = q1.sum(axis=1) + eps
    q1_out = q1 / q1_scaler[:, None]

    q2_scaler = q2.sum(axis=1) + eps
    q2_tmp = q2 / q2_scaler[:, None]
    q2_out = q2_tmp * q1_out[:, 1, None]

    q3_scaler = q3.sum(axis=1) + eps
    q3_tmp = q3 / q3_scaler[:, None]
    q3_out = q3_tmp * q2_out[:, 1, None]

    q4_scaler = q4.sum(axis=1) + eps
    q4_tmp = q4 / q4_scaler[:, None]
    q4_out = q4_tmp * q3_out.sum(axis=1)[:, None]

    q7_scaler = q7.sum(axis=1) + eps
    q7_tmp = q7 / q7_scaler[:, None]
    q7_out = q7_tmp * q1_out[:, 0, None]

    q9_scaler = q9.sum(axis=1) + eps
    q9_tmp = q9 / q9_scaler[:, None]
    q9_out = q9_tmp * q2_out[:, 0, None]

    q10_scaler = q10.sum(axis=1) + eps
    q10_tmp = q10 / q10_scaler[:, None]
    q10_out = q10_tmp * q4_out[:, 0, None]

    q11_scaler = q11.sum(axis=1) + eps
    q11_tmp = q11 / q11_scaler[:, None]
    q11_out = q11_tmp * q10_out.sum(axis=1)[:, None]

    q5_scaler = q5.sum(axis=1) + eps
    q5_tmp = q5 / q5_scaler[:, None]
    q5_out = q5_tmp * (q4_out[:, 1, None] + q11_out.sum(axis=1)[:, None])

    q6_scaler = q6.sum(axis=1) + eps
    q6_tmp = q6 / q6_scaler[:, None]
    q6_out = q6_tmp * (
        q5_out.sum(axis=1)[:, None] + q7_out.sum(axis=1)[:, None] + q9_out.sum(axis=1)[:, None]
    )

    q8_scaler = q8.sum(axis=1) + eps
    q8_tmp = q8 / q8_scaler[:, None]
    q8_out = q8_tmp * q6_out[:, 0, None]

    out = [
        q1_out,
        q2_out,
        q3_out,
        q4_out,
        q5_out,
        q6_out,
        q7_out,
        q8_out,
        q9_out,
        q10_out,
        q11_out,
    ]

    x = torch.cat(out, dim=1)
    return x


class Random(nn.Module):
    """Random predictor to benchmark against models."""

    def __init__(self, opt):
        super(Random, self).__init__()
        if opt.exp.task == "classification":
            self.n_classes = 5
            self.scale = nn.Softmax(dim=1)
        elif opt.exp.task == "regression":
            self.n_classes = 37
            self.scale = nn.Sigmoid(dim=1)
        self.batch_size = opt.compute.batch_size

    def forward(self, x=None):
        if x is not None:
            self.batch_size = x.shape[0]
        rand_pred = torch.rand((self.batch_size, self.n_classes))
        rand_pred = self.scale(rand_pred)
        rand_pred = rand_pred.cuda()
        return rand_pred


class EnsemblingTransforms:
    """Transforms for ensembling model's predictions"""

    def __init__(self):
        im_tf = [
            Rotation([0, 90, 180, -90]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=COLOR_JITTER_FACTOR,
                contrast=COLOR_JITTER_FACTOR,
                saturation=COLOR_JITTER_FACTOR,
                hue=COLOR_JITTER_FACTOR,
            ),
        ]
        self.im_tf = transforms.Compose(im_tf)

    def __call__(self, x):
        return self.im_tf(x)


class Rotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return tf.rotate(x, angle)
