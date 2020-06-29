from config import params
from torch import nn, optim
import os
from models import c3d,r3d,r21d,sscn
from datasets.predict_dataset import PredictDataset
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import random
import numpy as np
from tensorboardX import SummaryWriter
from visualze import *
import argparse
multi_gpu = 1;
start_epoch = 1
ckpt = None
params['batch_size'] = 8
params['num_workers'] = 4
params['dataset'] = '/home/Dataset/UCF-101-origin'
params['data']='UCF-101'
train_epoch = 300
learning_rate = 0.01

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
class Motion_MSEloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,output,clip_label,motion_mask):
        z = torch.pow((output-clip_label),2)
        loss = torch.mean(motion_mask*z)
        return loss

class Motion_MSEloss_NoFakeGt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, clip_label, motion_mask, recon_flags):
        loss_batch_list = []
        batch_size = output.size(0)
        for i in range(batch_size):
            Tinds = torch.nonzero(recon_flags[i]).squeeze()
            z = torch.pow((output[i][:,Tinds,:,:] - clip_label[i][:,Tinds,:,:]), 2)
            loss_i = torch.mean(z*motion_mask[i][:,Tinds,:,:])
            loss_batch_list.append(loss_i)
        loss_batch = torch.stack(loss_batch_list)
        loss = torch.mean(loss_batch)
        return loss
    
def train(train_loader,model,criterion_MSE,criterion_CE,optimizer,epoch,writer,root_path=None):
    torch.set_grad_enabled(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_recon = AverageMeter()
    losses_class = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    end = time.time()
    model.train()

    total_cls_loss = 0.0
    correct_cnt = 0
    total_cls_cnt = torch.zeros(4)
    correct_cls_cnt = torch.zeros(4)

    for step,(sample_clip,recon_clip,step_label,recon_rate,motion_mask, recon_flags) in enumerate(train_loader):
        data_time.update(time.time() - end)

        clip_input = sample_clip.cuda()
        clip_label = recon_clip.cuda()
        step_label = step_label.cuda()
        recon_rate = recon_rate.cuda()
        motion_mask = motion_mask.cuda()
        recon_flags = recon_flags.cuda()

        clip_output, step_output = model(clip_input)
        # loss_recon = criterion_MSE(clip_output, clip_label, motion_mask)
        loss_recon = criterion_MSE(clip_output, clip_label, motion_mask, recon_flags)
        loss_class = criterion_CE(step_output, step_label)
        loss = loss_recon + loss_class*0.1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_recon.update(loss_recon.item(), clip_input.size(0))
        losses_class.update(loss_class.item(), clip_input.size(0))
        losses.update(loss.item(), clip_input.size(0))
        batch_time.update(time.time()-end)
        end=time.time()

        prec_class = accuracy(step_output.data, step_label, topk=(1,))[0]
        acc.update(prec_class.item(), clip_input.size(0))
        total_cls_loss += loss_class.item()
        pts = torch.argmax(step_output, dim=1)
        correct_cnt += torch.sum(step_label == pts).item()
        for i in range(step_label.size(0)):
            total_cls_cnt[step_label[i]] += 1
            if step_label[i] == pts[i]:
                correct_cls_cnt[pts[i]] += 1

        if (step + 1)%params['display'] == 0:
            #showoneofthem(clip_label[0],root_path,epoch,step,'gt_s{}_r{}'.format(step_label[0].item(),recon_rate[0].item()))
            #showoneofthem(clip_output[0],root_path,epoch,step,'pd_s{}_r{}'.format(step_label[0].item(),recon_rate[0].item()))
            print('-----------------------------------------------')
            # for param in optimizer.param_groups:
            #     print("lr:",param['lr'])
            print("conv_lr:{} fc8_lr:{}".format(optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr']))

            p_str = "Epoch:[{0}][{1}/{2}]".format(epoch,step+1,len(train_loader));
            print(p_str)

            p_str = "data_time:{data_time:.3f},batch time:{batch_time:.3f}".format(data_time=data_time.val,batch_time=batch_time.val)
            print(p_str)

            p_str = "loss:{loss:.5f} loss_recon:{loss_recon:.5f} loss_class:{loss_class:.5f}".format(loss=losses.avg,
                                                                                                     loss_recon=losses_recon.avg,
                                                                                                     loss_class=losses_class.avg)
            print(p_str)

            p_str = "accuracy:{acc:.3f}".format(acc=acc.avg)
            print(p_str)

            total_step = (epoch-1)*len(train_loader) + step + 1
            info = {
                'loss': losses.avg,
                'loss_res': losses_recon.avg,
                'loss_cls': losses_class.avg*0.1
            }
            writer.add_scalars('train/loss', info, total_step)

            info_acc = {}
            for cls in range(correct_cls_cnt.size(0)):
                acc_cls = correct_cls_cnt[cls]/total_cls_cnt[cls]
                info_acc['cls{}'.format(cls)] = acc_cls
            info_acc['avg'] = acc.avg*0.01
            writer.add_scalars('train/acc', info_acc, total_step)

    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_acc = correct_cnt / len(train_loader.dataset)
    print('[TRAIN] loss_cls: {:.3f}, acc: {:.3f}'.format(avg_cls_loss, avg_acc))
    print(correct_cls_cnt)
    print(total_cls_cnt)
    print(correct_cls_cnt / total_cls_cnt)


def validation(val_loader,model,criterion_MSE,criterion_CE,optimizer,epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_recon = AverageMeter()
    losses_class = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    model.eval()
    end = time.time()
    total_loss = 0.0

    total_cls_loss = 0.0
    correct_cnt = 0
    total_cls_cnt = torch.zeros(4)
    correct_cls_cnt = torch.zeros(4)

    with torch.no_grad():
        for step, (sample_clip, recon_clip, step_label, recon_rate,motion_mask,recon_flags) in enumerate(val_loader):
            data_time.update(time.time() - end)

            clip_input = sample_clip.cuda()
            clip_label = recon_clip.cuda()
            step_label = step_label.cuda()
            recon_rate = recon_rate.cuda()
            motion_mask = motion_mask.cuda()
            recon_flags = recon_flags.cuda()

            clip_output, step_output = model(clip_input)
            # loss_recon = criterion_MSE(clip_output, clip_label, motion_mask)
            loss_recon = criterion_MSE(clip_output, clip_label, motion_mask, recon_flags)
            loss_class = criterion_CE(step_output, step_label)
            loss = loss_recon + loss_class*0.1

            losses_recon.update(loss_recon.item(), clip_input.size(0))
            losses_class.update(loss_class.item(), clip_input.size(0))
            losses.update(loss.item(), clip_input.size(0));
            batch_time.update(time.time()-end);
            end = time.time();
            total_loss +=loss.item()

            prec_class = accuracy(step_output.data, step_label, topk=(1,))[0]
            acc.update(prec_class.item(), clip_input.size(0))
            total_cls_loss += loss_class.item()
            pts = torch.argmax(step_output, dim=1)
            correct_cnt += torch.sum(step_label == pts).item()
            for i in range(step_label.size(0)):
                total_cls_cnt[step_label[i]] += 1
                if step_label[i] == pts[i]:
                    correct_cls_cnt[pts[i]] += 1

            if (step +1) % params['display'] == 0:
                print('-----------------------------validation-------------------')
                p_str = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_loader))
                print(p_str)

                p_str = 'data_time:{data_time:.3f},batch time:{batch_time:.3f}'.format(data_time=data_time.val,batch_time=batch_time.val);
                print(p_str)

                p_str = "loss:{loss:.5f} loss_recon:{loss_recon:.5f} loss_class:{loss_class:.5f}".format(loss=losses.avg,
                                                                                                         loss_recon=losses_recon.avg,
                                                                                                         loss_class=losses_class.avg)
                print(p_str)

                p_str = "accuracy:{acc:.3f}".format(acc=acc.avg)
                print(p_str)

    avg_cls_loss = total_cls_loss / len(val_loader)
    avg_acc = correct_cnt / len(val_loader.dataset)
    print('[VAL] loss_cls: {:.3f}, acc: {:.3f}'.format(avg_cls_loss, avg_acc))
    print(correct_cls_cnt)
    print(total_cls_cnt)
    print(correct_cls_cnt / total_cls_cnt)

    avg_loss = losses.avg
    return avg_loss;


def load_pretrained_weights(ckpt_path):
    adjusted_weights = {};
    pretrained_weights = torch.load(ckpt_path, map_location='cpu');
    for name, params in pretrained_weights.items():
        if "module" in name:
            name = name[name.find('.') + 1:]
        adjusted_weights[name] = params;
    return adjusted_weights;

def parse_args():
    parser = argparse.ArgumentParser(description='Video Clip Restruction and Playback Rate Prediction')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
    parser.add_argument('--model_name', type=str, default='c3d', help='model name')
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(vars(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.model_name == 'c3d':
        model=c3d.C3D(with_classifier=False)
    elif args.model_name == 'r3d':
        model=r3d.R3DNet((1,1,1,1),with_classifier=False)
    elif args.model_name == 'r21d':
        model=r21d.R2Plus1DNet((1,1,1,1),with_classifier=False)
    print(args.model_name)
    model = sscn.SSCN_OneClip(base_network=model, with_classifier=True, num_classes=4)
    if ckpt:
        weight = load_pretrained_weights(ckpt)
        model.load_state_dict(weight, strict=True)
    #train
    train_dataset =PredictDataset(params['dataset'],mode="train",args=args);
    # if params['data'] =='kinetics-400':
    #     val_dataset = PredictDataset(params['dataset'],mode='val',args=args);
    if params['data'] == 'UCF-101':
        val_size = 800
        train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset) - val_size, val_size))
    elif params['data'] == 'hmdb':
        val_size = 400
        train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset) - val_size, val_size))

    train_loader = DataLoader(train_dataset,batch_size=params['batch_size'],shuffle=True,num_workers=params['num_workers'],drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=params['batch_size'],shuffle=True,num_workers=params['num_workers'],drop_last=True)
    if multi_gpu ==1:
        model = nn.DataParallel(model)
    model = model.cuda()
    criterion_CE = nn.CrossEntropyLoss().cuda()
    # criterion_MSE = Motion_MSEloss().cuda()
    criterion_MSE = Motion_MSEloss_NoFakeGt().cuda()

    model_params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'fc8' in key:
                print(key)
                model_params += [{'params':[value],'lr':10*learning_rate}]
            else:
                model_params += [{'params':[value],'lr':learning_rate}]
    optimizer = optim.SGD(model_params, momentum=params['momentum'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=50, factor=0.1)

    save_path = params['save_path_base'] + "train_predict_{}_".format(args.exp_name) + params['data']
    model_save_dir = os.path.join(save_path,time.strftime('%m-%d-%H-%M'))
    writer = SummaryWriter(model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    prev_best_val_loss = 100
    prev_best_loss_model_path = None
    for epoch in tqdm(range(start_epoch,start_epoch+train_epoch)):
        train(train_loader,model,criterion_MSE,criterion_CE,optimizer,epoch,writer,root_path=model_save_dir)
        val_loss = validation(val_loader,model,criterion_MSE,criterion_CE,optimizer,epoch)
        if val_loss < prev_best_val_loss:
            model_path = os.path.join(model_save_dir, 'best_model_{}.pth.tar'.format(epoch))
            torch.save(model.state_dict(), model_path)
            prev_best_val_loss = val_loss;
            if prev_best_loss_model_path:
                os.remove(prev_best_loss_model_path)
            prev_best_loss_model_path = model_path
        scheduler.step(val_loss);

        if epoch % 20 == 0:
            checkpoints = os.path.join(model_save_dir, 'model_{}.pth.tar'.format(epoch))
            torch.save(model.state_dict(),checkpoints)
            print("save_to:",checkpoints);


if __name__ == '__main__':
    seed = 632;
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()

