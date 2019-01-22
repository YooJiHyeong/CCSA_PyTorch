import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Network
from dataloader import TrainSet, TestSet


domain_adaptation_task = 'MNIST_to_USPS'
sample_per_class = 7
repetition = 0

batch = 256
epochs = 50

alpha = 0.25
print("alpha : ", alpha)

train_set = TrainSet(domain_adaptation_task, repetition, sample_per_class)
train_set_loader = DataLoader(train_set, batch_size=batch, shuffle=True, drop_last=True)
test_set = TestSet(domain_adaptation_task, repetition, sample_per_class)
test_official_loader = DataLoader(test_set, batch_size=batch, shuffle=True, drop_last=True)
print("Dataset Length Train : ", len(train_set), " Test : ", len(test_set))

device = torch.device("cuda")
net = Network().to(device)
ce_loss = nn.CrossEntropyLoss()
optim = torch.optim.Adadelta(net.parameters())


# Constrastive Semantic Alignment Loss
def csa_loss(x, y, class_eq):
    margin = 1
    dist = F.pairwise_distance(x, y)
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()

def train(net, loader):
    net.train()
    # loss = model.train_on_batch([X1, X2], [y1, yc])
    for i, (src_img, src_label, target_img, target_label) in enumerate(loader):
        src_img, target_img = (x.to(device, dtype=torch.float) for x in [src_img, target_img])
        src_label, target_label = (x.to(device, dtype=torch.long) for x in [src_label, target_label])
        src_pred, src_feature = net(src_img)
        _, target_feature = net(target_img)
        
        ce  = ce_loss(src_pred, src_label)
        csa = csa_loss(src_feature, target_feature,
                       (src_label == target_label).float())
                                                    
        loss = (1 - alpha) * ce + alpha * csa 
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % 100 == 0:
            print("loss : %.4f"%(loss.item()))
    """
    # loss2 = model.train_on_batch([X2, X1], [y2, yc])    
    for i, (target_img, target_label, src_img, src_label) in enumerate(loader):
        src_img, target_img = list(map(lambda x:x.to(device, dtype=torch.float), [src_img, target_img]))
        src_label, target_label = list(map(lambda x:x.to(device, dtype=torch.long), [src_label, target_label]))
        src_pred, src_feature = net(src_img)
        _, target_feature = net(target_img)
        
        ce  = ce_loss(src_pred, src_label)
        csa = csa_loss(src_feature, target_feature,
                       (src_label == target_label).float())
                                                    
        loss = (1 - alpha) * ce + alpha * csa 
        optim.zero_grad()
        loss.backward()
        optim.step()
        if False: # i % 30 == 0:
            print("loss : %.4f"%(loss.item()))
    """
    return loss.item()

def test(net, loader):
    correct = 0
    net.eval()
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            pred, _ = net(img)
            _, idx = pred.max(dim=1)
            correct += (idx == label).sum().cpu().item()
    acc = correct / len(loader.dataset)
    return acc


for epoch in range(epochs):
    print("Epoch %d"%(epoch))
    train_loss = train(net, train_set_loader)
    test_acc = test(net, test_official_loader)
    print("Epoch[%d] acc : %.4f"%(epoch, test_acc))
