from __future__ import print_function
import numpy as np
import os
import torch
from torch.autograd import Variable

test_best_acc = 0.0


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=80,
                     init_lr=1e-3, init_wd=5e-3):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    wd = init_wd * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
        print('weight decay is set to {}'.format(wd))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = wd

    return optimizer


def train_epoch(epoch, model_name, model, data_loader, optimizer, criterion):
    model.train()
    use_cuda = torch.cuda.is_available()
    train_loss = 0
    correct = 0
    total = 0
    train_loss_all = []
    train_acc_all = []
    optimizer = exp_lr_scheduler(optimizer, epoch)
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        train_acc = 100.*correct/total
        train_err = 100. - train_acc
        if batch_idx % 15 == 14:
            train_loss_all.append(train_loss/(batch_idx+1))
            train_acc_all.append(100.*correct/total)
            print('{}: Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f} | Accuracy: {:.2f}% | Error: {:.2f}%'.format(
                model_name, epoch, (batch_idx+1)*len(inputs), len(data_loader.dataset),
                100.*(batch_idx+1)/len(data_loader), train_loss/(batch_idx+1), train_acc, train_err))

    return train_loss_all, train_acc_all


def test_epoch(epoch, model_name, model, data_loader, criterion):
    model.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    total = 0
    pred_all = np.empty(0)
    true_all = np.empty(0)
    global test_best_acc
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs, volatile=True), Variable(labels)
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).data[0]
        # get the index of the max log-probability
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        true_all = np.concatenate((true_all, labels.data.cpu().numpy().reshape(-1)))
        pred_all = np.concatenate((pred_all, predicted.cpu().numpy().reshape(-1)))
    test_loss /= len(data_loader)
    test_acc = 100.*correct / total
    test_err = 100. - test_acc
    true_all = np.array(true_all).reshape(-1, 1)
    pred_all = np.array(pred_all).reshape(-1, 1)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%, Error: {:.2f}%'.format(test_loss, test_acc, test_err))
    if test_acc > test_best_acc:
        test_best_acc = test_acc
        state = {
            'model_name': model_name+'_'+str(int(test_best_acc)),
            'model_param': model.state_dict(),
            'test_best_acc': test_best_acc,
            'epoch': epoch
        }
        if not os.path.isdir('../checkpoint'):
            os.mkdir('../checkpoint')
        torch.save(state, '../checkpoint/%s_%i_ckpt.t7' % (model_name, test_best_acc))
    test_best_err = 100 - test_best_acc
    print('Saving best accuracy({:.2f}%), minimal error({:.2f})...\n'.format(test_best_acc, test_best_err))
    return test_loss, test_acc, test_best_acc

