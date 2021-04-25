# Feel free to modifiy this file.

from torchvision import models, transforms

team_id = 9
team_name = "Bai Ze"
email_address = "team_leader_nyu_email_address@nyu.edu"

def get_model():
    return models.resnet18(num_classes=800)

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])