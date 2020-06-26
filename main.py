from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import math
import data_loader
from ResNet import MYNetA as model1
from ResNet import MYNetB as model2
from ResNet import MYNetC as model3
#from ResNet import MyEnsemble as MyEnsemble
import matplotlib.pyplot as plt




os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=3, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=5e-4,
                    help='the L2  weight decay')
parser.add_argument('--save_path', type=str, default="./tmp/origin_",
                    help='the path to save the model')
parser.add_argument('--root_path', type=str, default="OFFICE31/",
                    help='the path to load the data')
parser.add_argument('--source_dir', type=str, default="dslr",
                    help='the name of the source dir')
parser.add_argument('--test_dir', type=str, default="webcam",
                    help='the name of the test dir')
parser.add_argument('--diff_lr', type=bool, default=True,
                    help='the fc layer and the sharenet have different or same learning rate')
parser.add_argument('--gamma', type=int, default=1,
                    help='the fc layer and the sharenet have different or same learning rate')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def load_data():
    source_train_loader = data_loader.load_training(args.root_path, args.source_dir, args.batch_size, kwargs)
    target_train_loader = data_loader.load_training(args.root_path, args.test_dir, args.batch_size, kwargs)
    target_test_loader = data_loader.load_testing(args.root_path, args.test_dir, args.batch_size, kwargs)

    return source_train_loader, target_train_loader, target_test_loader

def train(epoch, model, source_loader, target_loader):
   #The last fully connected layer learning rate is 10 times the previous
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / args.epochs), 0.75)
    print("learning rate：", LEARNING_RATE)
    if args.diff_lr and model==model1:
        optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.Inception1.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.l2_decay)
    elif args.diff_lr and model==model2:
        optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.Inception2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.l2_decay)
    elif args.diff_lr and model==model3:
        optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.Inception3.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.l2_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=args.momentum,weight_decay = args.l2_decay)
 
    model.train()
    tgt_iter = iter(target_loader)
    for batch_idx, (source_data, source_label) in enumerate(source_loader):
        try:
            target_data, _ = tgt_iter.next()
        except Exception as err:
            tgt_iter=iter(target_loader)
            target_data, _ = tgt_iter.next()
        
        if args.cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        optimizer.zero_grad()

        #s_output, mmd_loss = model(source_data, target_data, source_label)
        s_output, loss = model(source_data, target_data, source_label)
        soft_loss = F.nll_loss(F.log_softmax(s_output, dim=1), source_label)
        
        # print((2 / (1 + math.exp(-10 * (epoch) / args.epochs)) - 1))
        if args.gamma == 1:
            gamma = 2 / (1 + math.exp(-10 * (epoch) / args.epochs)) - 1
        if args.gamma == 2:
            gamma = epoch /args.epochs
        #loss = soft_loss + gamma * mmd_loss
        loss = soft_loss + gamma * (loss)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlabel_Loss: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(source_data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), soft_loss.item(), loss.item()))
    
        # my append 
        if model == model1:
            epoch_plot.append(epoch)
            loss_mmd_plot.append(loss)
        if model == model2:
            epoch_plot.append(epoch)
            loss_lowrank_plot.append(loss)
        if model == model3:
            epoch_plot.append(epoch)
            loss_coral_plot.append(loss)

    return epoch_plot, loss_mmd_plot, loss_lowrank_plot, loss_coral_plot


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            s_output, t_output = model(data, data, target)
            
            '''
            if model == model1:
                s_output1.append(s_output)
            if model == model2:
                s_output2.append(s_output)
            if model == model3:
                s_output3.append(s_output)
            '''
            
            test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, reduction='sum').item()# sum up batch loss
            pred = s_output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum() 
            
        test_loss /= len(test_loader.dataset)
        print(args.test_dir, '\nTest set: Average loss: {:.4f}, Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    
    #return correct, pred, s_output1, s_output2, s_output3
    return correct, pred, s_output



def test_MyEnsemble(test_loader, out1, out2, out3):
    '''
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = MyEnsemble(predict1, predict2, predict3, target)
            print(output)
            pred = torch.argmax(output, 1)
            #test_loss += F.nll_loss(F.log_softmax(output, dim = 1), target, reduction='sum').item()# sum up batch loss
            #pred = output.data.max(1)[1] # get the index of the max log-probability
            print(pred)
            #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            correct += pred.eq(target.data).cpu().sum()


        #test_loss /= len(test_loader.dataset)

        #print(args.test_dir, '\nTest set: Average loss: {:.4f}, Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            #test_loss, correct, len(test_loader.dataset),
           # 100. * correct / len(test_loader.dataset)))
        print(args.test_dir, '\nTest set: Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    '''
    '''
    models_ensemble = [out1.cuda(), out2.cuda(), out3.cuda()]
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                images, labels = data.cuda(), target.cuda()
            predictions = [i for i in models_ensemble]
            avg_predictions = torch.mean(torch.stack(predictions), dim=0)
            _, predicted = torch.max(avg_predictions, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print('accuracy = {:f}'.format(100. * correct / total))
        print('correct: {:d}  total: {:d}'.format(correct, total))
    return correct, predicted

    '''



#********************************************************


   
if __name__ == '__main__':
    loss_mmd_plot = []
    loss_lowrank_plot = []
    loss_coral_plot = []
    test_acc_1 = []
    test_acc_2 = []
    test_acc_3 = []
    test_acc_4 = []
    pred1 = []
    pred2 = []
    pred3 = []
    pred4 = []
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    s_output1 = []
    s_output2 = []
    s_output3 = []
    model1 = model1(num_classes=31)
    model2 = model2(num_classes=31)
    model3 = model3(num_classes=31)
    #MyEnsemble = MyEnsemble(num_classes=31)
    print(model1)
    print(model2)
    print(model3)
    #print(MyEnsemble)

    if args.cuda:
        model1.cuda()
        model2.cuda()
        model3.cuda()
        #MyEnsemble.cuda()
    
    train_loader, unsuptrain_loader, test_loader = load_data()

    
    for M in range (1, 5):
        if M == 1:
            predict1 = []
            epoches = []
            epoch_plot = []
            for epoch in range(1, args.epochs + 1):
                epoch_plot, loss_mmd_plot, loss_lowrank_plot, loss_coral_plot = train(epoch, model1, train_loader, unsuptrain_loader)
                t_correct1, pred1, out1 = test(model1, test_loader)
                predict1.append(pred1)
                if t_correct1 > correct1:
                    correct1 = t_correct1
                print("%s max correct:" % args.test_dir, correct1.item())
                print(args.source_dir, "to", args.test_dir)
                
    
                # my append
                epoches.append(epoch)
                test_acc1 = (100. * correct1 / len(test_loader.dataset))
                test_acc_1.append(test_acc1) # append testing accuracy

            print('test accuracy for mmd=', test_acc1)
        
        
        if M == 2:
            predict2 = []
            epoches = []
            epoch_plot = []
            for epoch in range(1, args.epochs + 1):
                train(epoch, model2, train_loader, unsuptrain_loader)
                t_correct2, pred2, out2 = test(model2, test_loader)
                predict2.append(pred2)
                if t_correct2 > correct2:
                    correct2 = t_correct2
                print("%s max correct:" % args.test_dir, correct2.item())
                print(args.source_dir, "to", args.test_dir)
                
    
                # my append
                epoches.append(epoch)
                test_acc2 = (100. * correct2 / len(test_loader.dataset))
                test_acc_2.append(test_acc2) # append testing accuracy

            print('test accuracy for lowrank=', test_acc2)
            
        
        if M == 3:
            predict3 = []
            epoches = []
            epoch_plot = []
            for epoch in range(1, args.epochs + 1):
                train(epoch, model3, train_loader, unsuptrain_loader)
                t_correct3, pred3, out3 = test(model3, test_loader)
                predict3.append(pred3)
                if t_correct3 > correct3:
                    correct3 = t_correct3
                print("%s max correct:" % args.test_dir, correct3.item())
                print(args.source_dir, "to", args.test_dir)
                
    
                # my append
                epoches.append(epoch)
                test_acc3 = (100. * correct3 / len(test_loader.dataset))
                test_acc_3.append(test_acc3) # append testing accuracy

            print('test accuracy for coral=', test_acc3)
            
            
        if M == 4:
            '''
            #for epoch in range(1, args.epochs + 1):       
                t_correct4, pred4 = test_MyEnsemble(test_loader, out1, out2, out3)
                t_correct4 = torch.tensor(t_correct4)
                correct4 = torch.tensor(correct4)
                if t_correct4 > correct4:
                    correct4 = t_correct4
                print("%s max correct:" % args.test_dir, correct4.item())
                print(args.source_dir, "to", args.test_dir)
                
    
                # my append
                #epoches.append(epoch)
                test_acc4 = (100. * correct4 / len(test_loader.dataset))
                test_acc_4.append(test_acc4) # append testing accuracy

            #print('test accuracy for stack=', test_acc4)
            
            '''
  
    print('test accuracy for mmd=', test_acc1, 'test accuracy for lowrank=', test_acc2, 'test accuracy for coral=', test_acc3)
    
    
    
    # my plotting
    curve1, = plt.plot(epoch_plot, loss_mmd_plot, label='mmd')
    curve2, = plt.plot(epoch_plot, loss_lowrank_plot, label='low rank')
    curve3, = plt.plot(epoch_plot, loss_coral_plot, label='coral')
    #plt.title("Loss Curve for {})".format(model))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(handles=[curve1, curve2, curve3])
    plt.show()
    line1, = plt.plot(epoch_plot, test_acc_1, label='mmd')
    line2, = plt.plot(epoch_plot, test_acc_2, label='low rank')
    line3, = plt.plot(epoch_plot, test_acc_3, label='coral')
    #line4, = plt.plot(epoches, test_acc_4, label='ensemble')
    #plt.title("Accuracy Curve on test data for Source={} and Target={} (batch-size={}, lr={})".format(source1_name, target_name, batch_size, lr))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(handles=[line1, line2, line3])
    plt.show()
    