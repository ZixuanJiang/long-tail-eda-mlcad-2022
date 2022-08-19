import torch
import argparse
import dgl
import loss

import torch.nn.functional as F
from model import TimingGCNBatch
from CirDelay import CirDelay
from dgl.distributed import DistDataLoader
from dataloader import DistDataLoaderSampler

from sklearn.metrics import confusion_matrix, classification_report

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
    default=250,
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
drs_flag = args.two_stage == 'drs'
print("Start supervised training.")
print(args._get_args)

epoches = 350
batch_size = 10240
device = torch.device("cuda")

num_classes = 100
dataset = CirDelay(num_classes=num_classes)

model = TimingGCNBatch(dataset.num_node_features, dataset.num_edge_features, dataset.num_classes).to(device)

train_nid, test_nid = dataset.train_test_nids()
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
collator = dgl.dataloading.NodeCollator(dataset.graph(), train_nid, sampler)
dataloader = DistDataLoaderSampler(collator=collator, num_classes=dataset.num_classes, 
        batch_size=batch_size, shuffle=True, 
        drop_last=False, weighted_sampler=args.resampling,
        drs_flag=drs_flag, two_stage_start_epoch=args.two_stage_start_epoch)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200], gamma=0.1)

para_dict = {
    "num_classes": dataloader.num_classes,
    "num_class_list": dataloader.num_list,
    "device": device,
    "two_stage_drw": args.two_stage == 'drw',
    "two_stage_start_epoch": args.two_stage_start_epoch,
    "ce_label_smooth_epsilon": 0.5, # 0.1
    "ce_label_aware_smooth_head": 0.4,
    "ce_label_aware_smooth_tail": 0.0, # 0.1
    "ce_label_aware_smooth_shape": 'concave',
    "focal_loss_gamma": 4.0, # 2.0
    "class_balance_ce_beta": 0.9999, # 0.999
    "class_balance_focal_gamma": 1.0, # 1.0
    "class_balance_focal_beta": 0.9999, # 0.999
    "cost_sensitive_ce_gamma": 0.5, # 1.0, 0.5
    "LDAM_scale": 30.0,
    "LDAM_max_margin": 0.5,
    "SEQL_lambda": 0.05,
    "SEQL_gamma": 0.9,
    "CDT_gamma": 0.2, # 0.1
}
train_loss_function = getattr(loss, args.reweighting)(para_dict=para_dict).to(device)
#test_loss_function = torch.nn.CrossEntropyLoss().to(device)


def train(epoch):
    model.train()
    dataloader.update_sampler(epoch, epoches)
    train_loss_function.update(epoch)

    total_loss = 0
    total_nodes = 0

    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b.to(device) for b in blocks]
        input_features = blocks[0].srcdata['node_features']
        output_labels = blocks[-1].dstdata['label'].long()
        output_predictions = model(blocks, input_features)
        loss = train_loss_function(output_predictions, output_labels)
        total_loss += loss.item() * output_labels.shape[0]
        total_nodes += output_labels.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    print("epoch {}, train loss {:.5f}".format(epoch, total_loss/total_nodes))

def test_mode(mode='test'):
    if mode == 'test':
        nids = test_nid
    else:
        nids = train_nid
    test_collator = dgl.dataloading.NodeCollator(dataset.graph(), nids, sampler)
    test_dataloader = DistDataLoader(test_collator.dataset, collate_fn=test_collator.collate, 
        batch_size=batch_size, shuffle=False, drop_last=False)

    total = 0
    top_5_cor = 0
    top_1_cor = 0
    pre_all = []
    tru_all = []
    
    for input_nodes, output_nodes, blocks in test_dataloader:
        blocks = [b.to(device) for b in blocks]
        input_features = blocks[0].srcdata['node_features']
        truth = blocks[-1].dstdata['label'].long()
        output_predictions = model(blocks, input_features)

        _, pred = output_predictions.max(1)
        top_1_cor += (pred == truth).float().sum()

        pre_all.extend(pred.cpu().numpy())
        tru_all.extend(truth.cpu().numpy())

        _, pred = output_predictions.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(truth.view(1,-1).expand_as(pred))
        top_5_cor += correct[:5].reshape(-1).float().sum(0)
        total += truth.size(0)

    print("{}: Top1 Acc: {:.5f} Top5 Acc: {:.5f}".format(mode, top_1_cor/total, top_5_cor/total))

    return pre_all, tru_all
    
def test(epoch, confusion_matrix_flag=True):
    model.eval()
    if confusion_matrix_flag:
        train_pred = []
        train_true = []
        y_pred = []
        y_true = []

    with torch.no_grad():
        print('======= Training dataset ======')
        train_pred, train_true = test_mode('train')

        print('======= Test dataset ======')
        y_pred, y_true = test_mode('test')

        if confusion_matrix_flag:
            print(classification_report(train_true, train_pred, digits=4))
            print(classification_report(y_true, y_pred, digits=4))
        
for epoch in range(epoches):
    train(epoch)
    if epoch % 20 == 0 or epoch == epoches - 1:
        test(epoch, epoch == epoches - 1)
        #test(epoch)

if args.model_save_path is not None:
    torch.save(model.state_dict(), args.model_save_path)
