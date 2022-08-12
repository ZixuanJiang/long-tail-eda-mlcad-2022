import argparse
import torch
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report

import resnet
import loss
from wm_811k_dataset import WM811K

parser = argparse.ArgumentParser(description='Parameters for supervised training')
parser.add_argument(
    "--reweighting",
    help="Reweighting techniques. Select one from loss.py.",
    required=False,
    default="CrossEntropy",
    type=str,
)

parser.add_argument(
    "--resampling",
    help="Resampling techniques. Select one from [balance, square, progressive].",
    required=False,
    default=None,
    type=str,
)

parser.add_argument(
    "--two_stage",
    help="Deferred re-balancing by resampling (drs) and by re-weighting (drw).",
    required=False,
    default=None,
    type=str,
)

parser.add_argument(
    "--two_stage_start_epoch",
    help="The epoch when we start second stage training.",
    required=False,
    default=80,
    type=int,
)

parser.add_argument(
    "--model_save_path",
    help="If specified, the model will be saved",
    required=False,
    default=None,
    type=str,
)

args = parser.parse_args()
print("Start supervised training.")
print(args._get_args)

epoches = 100
step_per_epoch = 500
batch_size = 64
seed = 314
device = torch.device("cuda:0")

# If true, use the original train test split.
# Otherwise, split the original train set into a new train test split.
use_original_train_test_split = False
fraction_of_training_samples = 1

feature_network = resnet.ResNet(resnet.BasicBlock, [3, 3, 3], linear_head=True).to(device)

optimizer = torch.optim.SGD(feature_network.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)

normalize = transforms.Normalize((0.4474541319767604), (0.29148643315474054))
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=180),
    transforms.ToTensor(),
    normalize, ]
)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize, ]
)

drs_flag = args.two_stage == 'drs'
num_samples_original_trainset = 54355
if use_original_train_test_split:
    if fraction_of_training_samples == 1:
        num_train_samples = num_samples_original_trainset
        train_indices = None
    elif fraction_of_training_samples > 0 and fraction_of_training_samples < 1:
        num_train_samples = int(num_samples_original_trainset * fraction_of_training_samples)
        indices = torch.randperm(num_samples_original_trainset, generator=torch.Generator().manual_seed(seed)).tolist()
        train_indices = indices[0: num_train_samples]
    else:
        assert False, "invalid fraction_of_training_samples"

    trainset = WM811K(mode='train', dim=32, transform=train_transform, weighted_sampler=args.resampling, indices=train_indices, drs_flag=drs_flag, two_stage_start_epoch=args.two_stage_start_epoch)
    testset = WM811K(mode='test', dim=32, transform=test_transform)
    num_test_samples = testset.data.shape[0]
else:
    num_train_samples = int(num_samples_original_trainset * 0.8 * fraction_of_training_samples)
    num_test_samples = int(num_samples_original_trainset * 0.2)

    indices = torch.randperm(num_samples_original_trainset, generator=torch.Generator().manual_seed(seed)).tolist()
    train_indices = indices[0: num_train_samples]
    test_indices = indices[num_train_samples: num_train_samples + num_test_samples]

    trainset = WM811K(mode='train', dim=32, transform=train_transform, indices=train_indices, weighted_sampler=args.resampling, drs_flag=drs_flag, two_stage_start_epoch=args.two_stage_start_epoch)
    testset = WM811K(mode='train', dim=32, transform=test_transform, indices=test_indices)

print(f"training samples {num_train_samples}, testing samples {num_test_samples}")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

iter_trainloader = iter(trainloader)

para_dict = {
    "num_classes": trainset.num_classes,
    "num_class_list": trainset.num_list,
    "device": device,
    "two_stage_drw": args.two_stage == 'drw',
    "two_stage_start_epoch": args.two_stage_start_epoch,
    "ce_label_smooth_epsilon": 0.1,
    "ce_label_aware_smooth_head": 0.4,
    "ce_label_aware_smooth_tail": 0.1,
    "ce_label_aware_smooth_shape": 'concave',
    "focal_loss_gamma": 2.0,
    "class_balance_ce_beta": 0.999,
    "class_balance_focal_gamma": 1.0,
    "class_balance_focal_beta": 0.999,
    "cost_sensitive_ce_gamma": 1.0,
    "LDAM_scale": 30.0,
    "LDAM_max_margin": 0.5,
    "SEQL_lambda": 0.05,
    "SEQL_gamma": 0.9,
    "CDT_gamma": 0.2,
}
train_loss_function = getattr(loss, args.reweighting)(para_dict=para_dict).to(device)
test_loss_function = torch.nn.CrossEntropyLoss().to(device)


def train(epoch):
    feature_network.train()
    trainloader.dataset.update(epoch, epoches)
    train_loss_function.update(epoch)

    total_loss = 0
    for step_id in range(step_per_epoch):
        try:
            images, labels = iter_trainloader.next()
        except:
            iter_trainloader = iter(trainloader)
            images, labels = iter_trainloader.next()

        images, labels = images.to(device), labels.to(device)
        prediction = feature_network(images)
        loss = train_loss_function(prediction, labels)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f'epoch {epoch}, train loss {total_loss / step_per_epoch}, ', end='', flush=True)


def test(epoch, confusion_matrix_flag=False):
    feature_network.eval()
    total_loss = 0
    correct = 0
    if confusion_matrix_flag:
        y_pred = []
        y_true = []

    with torch.no_grad():
        for batch_id, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            prediction = feature_network(images)
            loss = test_loss_function(prediction, labels)
            total_loss += loss * images.shape[0]

            _, predicted_label = prediction.max(1)
            correct += predicted_label.eq(labels).sum().item()

            if confusion_matrix_flag:
                y_pred.extend(predicted_label.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

    print(f'test CE loss {total_loss / num_test_samples}, test accuracy {correct / num_test_samples}, correct {correct}', flush=True)
    if confusion_matrix_flag:
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, digits=4))


for epoch in range(epoches):
    train(epoch)
    last_epoch_flag = (epoch == epoches - 1)
    test(epoch, confusion_matrix_flag=last_epoch_flag)

if args.model_save_path is not None:
    torch.save(feature_network.state_dict(), args.model_save_path)
