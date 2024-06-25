import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
def optimizer_picker(optimization, param, lr):
    if optimization == 'adam':
        optimizer = torch.optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = torch.optim.SGD(param, lr=lr)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = torch.optim.Adam(param, lr=lr)
    return optimizer


def train_one_epoch(data_loader, model, criterion, optimizer, loss_mode, device):
    running_loss = 0
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x
        
        loss = criterion(output, batch_y)

        loss.backward()
        optimizer.step()
        running_loss += loss
    return {
            "loss": running_loss.item() / len(data_loader),
            }

def evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device, trigger_label):
    ta = eval(data_loader_val_clean, model, device, isPoisioned=0, print_perform=True, trigger_label=trigger_label)
    asr = eval(data_loader_val_poisoned, model, device, isPoisioned=1, print_perform=False, trigger_label=trigger_label)
    index_poison_success = list(set(ta.get("index")) & set(asr.get("index")))
    asr_new = len(index_poison_success)/len(ta.get("index"))
    return {
            'clean_acc': ta['acc'], 'clean_loss': ta['loss'],
            'asr': asr['acc'], 'asr_loss': asr['loss'],
            'asr_new': asr_new
            }

def eval(data_loader, model, device, batch_size=64, isPoisioned=0, print_perform=False, trigger_label=0):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    loss_sum = []
    count = 0
    index = []
    for (batch_x, batch_y) in tqdm(data_loader):

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        batch_y_predict = model(batch_x)[0]
        loss = criterion(batch_y_predict, batch_y)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)


        # save img
        # if isPoisioned==1:
        #     count = count + 1
        #     i = 0
        #     while i < len(batch_x):
        #         img, targetOriginal, target = batch_x[i], batch_y[i], batch_y_predict[i]
        #         if batch_y[i] != batch_y_predict[i]:
        #             savePath = '/home/shixiong22/code/Sel-CL-main/badnets/imgSave/6/' + str(batch_y[i].item()) + '--' + str(
        #                 batch_y_predict[i].item()) + '_' + str(count)+'_'+str(i) + '.jpg'
        #             save_image(img, savePath)
        #         i += 1
        y_true.append(batch_y)
        # calculate Attack Success Rate (ASR)
        if isPoisioned == 0:
            i = 0
            while i < len(batch_y_predict):
                if batch_y_predict[i].item() != trigger_label:
                    index.append(count * batch_size + i)
                i = i + 1
            count = count + 1
        if isPoisioned != 0:
            i = 0
            while i < len(batch_y_predict):
                if batch_y_predict[i].item() == trigger_label:
                    index.append(count * batch_size + i)
                i = i + 1
            count = count + 1

        y_predict.append(batch_y_predict)
        loss_sum.append(loss.item())

    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)
    loss = sum(loss_sum) / len(loss_sum)

    if print_perform:
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": loss,
            "index": index
            }


def eval_v2(data_loader, model, device, batch_size=64, print_perform=False):
    model.eval()
    loss_per_batch = []
    test_loss = 0
    correct_1 = 0
    correct_5 = 0
    y_true = []
    y_predict = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            try:
                output, _ = model(data)
            except:
                output = model(data)
            # output = torch.argmax(output, dim=1)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            result = accuracy_v3(output, target, top=[1, 5])
            correct_1 += result[0].item()
            correct_5 += result[1].item()
            # correct += pred.eq(target.view_as(pred)).sum().item()
            y_true.append(target)
            y_predict.append(output)
    test_loss /= len(data_loader.dataset)
    print('\nTest set prediction branch: Average loss: {:.4f}, top1 Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct_1, len(data_loader.dataset),
        100. * correct_1 / len(data_loader.dataset)))
    print('\nTest set prediction branch: Average loss: {:.4f}, top5 Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct_5, len(data_loader.dataset),
        100. * correct_5 / len(data_loader.dataset)))

    loss_per_epoch = np.average(loss_per_batch)
    acc_val_per_epoch = np.array(100. * correct_1 / len(data_loader.dataset))

    # return (loss_per_epoch, acc_val_per_epoch)
    # if print_perform:
    #     print(classification_report(target.cpu(), output.cpu(), target_names=data_loader.dataset.classes))
    return {
            "acc": acc_val_per_epoch,
            "loss": loss_per_epoch,
            }

def accuracy_v3(preds, labels, top=[1,5]):
    """Compute the precision@k for the specified values of k"""
    result = []
    maxk = max(top)
    batch_size = preds.size(0)

    _, pred = preds.topk(maxk, 1, True, True)
    pred = pred.t() # pred[k-1] stores the k-th predicted label for all samples in the batch.
    correct = pred.eq(labels.view(1,-1).expand_as(pred))

    for k in top:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k)

    return result