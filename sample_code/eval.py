# Please do not change this file.
# We will use this eval script to benchmark your model.
# If you find a bug, post it on campuswire.

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from dataloader import CustomDataset
from submission import get_model, eval_transform, team_id, team_name, email_address

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', type=str)
args = parser.parse_args()

evalset = CustomDataset(root='/dataset', split="val", transform=eval_transform)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=2)

net = get_model()
checkpoint = torch.load(args.checkpoint_path)
net.load_state_dict(checkpoint)
net = net.cuda()

net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in evalloader:
        images, labels = data

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print(f"Team {team_id}: {team_name} Accuracy: {(100 * correct / total):.2f}%")
print(f"Team {team_id}: {team_name} Email: {email_address}")