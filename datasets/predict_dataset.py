import os
import torch.utils.data as data
import cv2
import sys

sys.path.append('..')
import random
import skvideo.io
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
import pandas as pd
import argparse
import collections
from datasets import patch_region

envs = os.environ;


class PredictDataset(data.Dataset):
    def __init__(self, root, mode="train", args=None):

        self.transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.ToTensor()])

        self.root = root;
        self.mode = mode;
        self.args = args
        self.toPIL = transforms.ToPILImage()
        self.tensortrans = transforms.Compose([transforms.ToTensor()])

        self.split = '1'

        train_split_path = os.path.join(root, 'split', 'trainlist0' + self.split + '.txt')
        self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        test_split_path = os.path.join(root, 'split', 'testlist0' + self.split + '.txt')
        self.test_split = pd.read_csv(test_split_path, header=None)[0]
        if mode == 'train':
            self.list = self.train_split
        else:
            self.list = self.test_split

        self.batch_size = 8
        self.sample_step_list = [1, 2, 4, 8]
        self.recon_rate_list = [2]
        self.sample_retrieval = collections.OrderedDict()
        for i in range(len(self.recon_rate_list)):
            recon_rate = self.recon_rate_list[i]
            self.sample_retrieval[recon_rate] = self.sample_step_list[i:]
        self.recon_rate_label_batch = []
        recon_rate_label = np.random.randint(low=0, high=len(self.recon_rate_list))
        for i in range(self.batch_size):
            self.recon_rate_label_batch.append(recon_rate_label)

    def __getitem__(self, index):

        if len(self.recon_rate_label_batch) == 0:
            recon_rate_label = np.random.randint(low=0, high=len(self.recon_rate_list))
            for i in range(self.batch_size):
                self.recon_rate_label_batch.append(recon_rate_label)
        recon_rate_label = self.recon_rate_label_batch.pop(0)
        recon_rate = self.recon_rate_list[recon_rate_label]

        videodata, sample_step_label = self.loadcvvideo_Finsert(index, recon_rate=recon_rate, sample_step=None)
        target_clip = self.crop(videodata)
        sample_step = self.sample_step_list[sample_step_label]
        sample_inds = torch.arange(0, len(videodata), step=sample_step)
        sample_clip = target_clip[:, sample_inds, :, :]

        if sample_step > 1:
            recon_step = int(sample_step / recon_rate)
            recon_inds = torch.arange(0, len(videodata), step=recon_step)
            recon_clip = target_clip[:, recon_inds, :, :]
        else:
            recon_step = sample_step
            recon_inds = torch.arange(0, len(videodata), step=recon_step)
            recon_clip1 = target_clip[:, recon_inds, :, :]
            recon_clip2 = target_clip[:, recon_inds, :, :]
            for i in range(len(recon_inds) - 1):
                recon_clip2[:, i, :, :] = (recon_clip1[:, i, :, :] + recon_clip1[:, i + 1, :, :]) / 2
            c, t, h, w = recon_clip1.size()
            recon_clip = torch.cat((recon_clip1.unsqueeze(dim=2), recon_clip2.unsqueeze(dim=2)), dim=2).reshape(c,
                                                                                                                2 * t,
                                                                                                                h, w)
        recon_clip_mask_len = int(recon_clip.size(1) / sample_step)
        recon_clip_mask = recon_clip[:, :recon_clip_mask_len, :, :]
        #         patch_loss_mask_list = []
        motion_mask_list = []
        for i in range(sample_step):
            clip16 = target_clip[:, i * 16:(i + 1) * 16, :, :]
            max_patch_x, max_patch_y, patch_loss_mask = patch_region.getPatchLossMask_stepsize(clip16, recon_clip_mask)
            #             patch_loss_mask_list.append(patch_loss_mask)

            motion_mask = torch.ones_like(recon_clip_mask)
            patch_size = 28
            h = 112
            w = 112
            sigma = patch_size / 2
            ux = max_patch_x + patch_size / 2
            uy = max_patch_y + patch_size / 2
            vx = np.arange(0, h, 1)
            vx = np.reshape(vx, (h, 1))
            vy = np.arange(0, w, 1)
            vy = np.reshape(vy, (1, w))
            fx = np.exp((-0.5) * (1 / sigma) * (1 / sigma) * (vx - ux) * (vx - ux))
            fy = np.exp((-0.5) * (1 / sigma) * (1 / sigma) * (vy - uy) * (vy - uy))
            fxy = np.matmul(fx, fy)
            fxy = 1.2 / (fxy.max() - fxy.min()) * (fxy - fxy.min()) + 0.8
            fxy = torch.from_numpy(fxy).type(torch.FloatTensor)
            motion_mask = motion_mask[:, ] * fxy
            motion_mask_list.append(motion_mask)
        #         patch_loss_mask = torch.cat(patch_loss_mask_list,dim=1)
        motion_mask = torch.cat(motion_mask_list, dim=1)

        return sample_clip, recon_clip, sample_step_label, recon_rate, motion_mask

    def loadcvvideo_Finsert(self, index, recon_rate=None, sample_step=None):
        need = 16;
        fname = self.list[index]
        fname = os.path.join(self.root, 'video', fname)

        capture = cv2.VideoCapture(fname);
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if sample_step is None:
            if recon_rate:
                sample_step_proposal = self.sample_retrieval[recon_rate]
                proposal_idx = np.random.randint(low=0, high=len(sample_step_proposal))
                sample_step = sample_step_proposal[proposal_idx]
                sample_step_label = self.sample_step_list.index(sample_step)
            else:
                sample_step_label = np.random.randint(low=0, high=len(self.sample_step_list))
                sample_step = self.sample_step_list[sample_step_label]
        else:
            sample_step_label = self.sample_step_list.index(sample_step)


        sample_len = need * sample_step
        shortest_len = sample_len + 1
        while frame_count < shortest_len:
            index = np.random.randint(self.__len__());
            fname = self.list[index]
            fname = os.path.join(self.root, 'video', fname)

            capture = cv2.VideoCapture(fname);
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT));

        start = np.random.randint(0, frame_count - shortest_len + 1);
        if start > 0:
            start = start - 1;
        buffer = []
        count = 0;
        retaining = True;
        sample_count = 0

        while (sample_count < sample_len and retaining):
            retaining, frame = capture.read();

            if retaining is False:
                count += 1;

                break;
            if count >= start:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buffer.append(frame)
                sample_count = sample_count + 1
            count += 1;

        capture.release();
        while len(buffer) < sample_len:
            index = np.random.randint(self.__len__());
            print('retaining:{} buffer_len:{} sample_len:{}'.format(retaining, len(buffer), sample_len))
            # buffer,sampe_label = self.loadcvvideo_msr(index)
            buffer, sample_step_label = self.loadcvvideo_Finsert(index, recon_rate, sample_step)
            print('reload')

        return buffer, sample_step_label

    def crop(self, frames):
        video_clips = [];
        seed = random.random()

        for frame in frames:
            random.seed(seed)

            frame = self.toPIL(frame);

            frame = self.transforms(frame);

            video_clips.append(frame)

        clip = torch.stack(video_clips).permute(1, 0, 2, 3)

        return clip

    def __len__(self):

        return len(self.list)


class ClassifyDataSet(data.Dataset):
    def __init__(self, root, mode="train"):

        self.transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.RandomCrop(112),
            transforms.ToTensor()])

        self.root = root;
        self.mode = mode;
        self.videos = []
        self.labels = [];
        self.toPIL = transforms.ToPILImage()
        self.split = '1';

        class_idx_path = os.path.join(root, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]
        if self.mode == 'train':
            train_split_path = os.path.join(root, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root, 'split', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
        print('Use split' + self.split)

    def loadcvvideo(self, fname, count_need=16):
        fname = os.path.join(self.root, 'video', fname)
        capture = cv2.VideoCapture(fname);
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT));

        if count_need == 0:
            count_need = frame_count;
        start = np.random.randint(0, frame_count - count_need + 1);
        buffer = []
        count = 0;
        retaining = True;
        sample_count = 0;

        while (sample_count < count_need and retaining):
            retaining, frame = capture.read();

            if retaining is False:
                count += 1;

                break;
            if count >= start:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = cv2.resize(frame, (171, 128))
                buffer.append(frame)
                sample_count = sample_count + 1
            count += 1;

        capture.release();

        return buffer, retaining

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_split) - 1
        else:
            return len(self.test_split) - 1

    def __getitem__(self, index):
        if self.mode == 'train':
            videoname = self.train_split[index]
        else:
            videoname = self.test_split[index]

        if self.mode == 'train':
            videodata, retrain = self.loadcvvideo(videoname, count_need=16)
            while retrain == False or len(videodata) < 16:
                print('reload');
                index = np.random.randint(self.__len__());

                videoname = self.train_split[index]
                videodata, retrain = self.loadcvvideo(videoname, count_need=16)

            videodata = self.randomflip(videodata)

            video_clips = [];
            seed = random.random()

            for frame in videodata:
                random.seed(seed)

                frame = self.toPIL(frame);

                frame = self.transforms(frame);

                video_clips.append(frame)

            clip = torch.stack(video_clips).permute(1, 0, 2, 3)

        elif self.mode == 'test':
            videodata, retrain = self.loadcvvideo(videoname, count_need=0)
            while retrain == False or len(videodata) < 16:
                print('reload');
                index = np.random.randint(self.__len__());

                videoname = self.test_split[index]
                videodata, retrain = self.loadcvvideo(videoname, count_need=16)
            clip = self.gettest(videodata);
        label = self.class_label2idx[videoname[:videoname.find('/')]]

        return clip, label - 1

    def randomflip(self, buffer):
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer;

    def gettest(self, videodata):
        length = len(videodata)

        all_clips = []

        for i in np.linspace(8, length - 8, 10):
            clip_start = int(i - 8)
            clip = videodata[clip_start: clip_start + 16]
            trans_clip = []
            # fix seed, apply the sample `random transformation` for all frames in the clip
            seed = random.random()
            for frame in clip:
                random.seed(seed)
                frame = self.toPIL(frame)  # PIL image
                frame = self.transforms(frame)  # tensor [C x H x W]
                trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
            clip = torch.stack(trans_clip).permute([1, 0, 2, 3])

            all_clips.append(clip)
        return torch.stack(all_clips)


def parse_args():
    parser = argparse.ArgumentParser(description='Video Clip Restruction and Order Prediction')
    parser.add_argument('--lpls', type=bool, default=False, help='use lpls_loss or not')
    parser.add_argument('--msr', type=bool, default=False, help='use multi sample rate or not')
    parser.add_argument('--vcop', type=bool, default=True, help='predict video clip order or not')
    parser.add_argument('--num_order', type=int, default=2, help='number of video clip order to predict')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    com = PredictDataset('/home/yaoyuan/Workspace/Dataset/UCF-101-origin', mode="train", args=args);
    train_dataloader = DataLoader(com, batch_size=8, num_workers=4, shuffle=True, drop_last=True)
    for i, (clip1, clip2, a, b) in enumerate(train_dataloader):
        print(i)
        print(clip1.size())
        print(clip2.size())
        # print(a.size())
        # print(b.size())
        print(a)
        print(b)

        # a=1
        # a[1]
        # print(label)











