# Feel free to modifiy this file.

from torchvision import models, transforms

team_id = 1
team_name = "your_team_name"
email_address = "team_leader_nyu_email_address@nyu.edu"

def get_model():
    return models.resnet18(num_classes=800)

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])