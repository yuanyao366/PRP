import torch.nn  as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch


patch_size =28;

patch_col  = 112/28

patch_num = (112*112)/(28*28)
#clip 3*16*112*112
def getVarious(clip):
    img1=clip[:,0,::]
    img2=clip[:,5,::]
    img3=clip[:,10,::]
    img4=clip[:,15,::]
    imgs = [img1,img2,img3,img4]
    location  = getPatchLocation_stepsize(imgs)
    # print("lll",location)
    return location

def getPatchLocation(imgs):
    a,b,c,d = imgs

    max_various = 0;
    max_location = 0;
    patch_location =0;
    criterion = nn.MSELoss().cuda()

    while patch_location < patch_num:
        patch_x_id=int(patch_location/patch_col)
        patch_y_id=int(patch_location%patch_col)
        patch_x = patch_x_id*patch_size
        patch_y = patch_y_id * patch_size
        a_patch = a[:,patch_x:patch_x+patch_size,patch_y:patch_y+patch_size]
        b_patch = b[:,patch_x:patch_x+patch_size,patch_y:patch_y+patch_size]
        c_patch = c[:,patch_x:patch_x+patch_size,patch_y:patch_y+patch_size]
        d_patch = d[:,patch_x:patch_x+patch_size,patch_y:patch_y+patch_size]
        loss1 = criterion(a_patch,b_patch)
        loss2 = criterion(b_patch,c_patch)
        loss3 = criterion(c_patch,d_patch)
        loss = loss1+loss2+loss3;
        if loss >= max_various:
            max_location = patch_location;
            max_various = loss;
        patch_location = patch_location+1
    max_x_id=int(max_location/patch_col)
    max_y_id=int(max_location%patch_col)
    max_patch_x = max_x_id*patch_size
    max_patch_y = max_y_id * patch_size

    a_patch = a[:,max_patch_x:max_patch_x+patch_size,max_patch_y:max_patch_y+patch_size]
    image = a_patch.detach()
    image = image.cpu().numpy()
    print(image.shape)
    image = image.transpose(1, 2, 0)
        # image = image * 255
    plt.imshow(image)
    plt.show();
    return max_patch_x,max_patch_y;

def getPatchLocation_stepsize(imgs):
    a,b,c,d = imgs

    max_various = 0;
    step_size = int(patch_size/4)
    max_patch_x = 0
    max_patch_y = 0
    criterion = nn.MSELoss().cuda()
    current_x = 0;
    while current_x < 112-patch_size:
        current_y = 0;
        while current_y < 112-patch_size:
            a_patch = a[:, current_x:current_x + patch_size, current_y:current_y + patch_size]
            b_patch = b[:, current_x:current_x + patch_size, current_y:current_y + patch_size]
            c_patch = c[:, current_x:current_x + patch_size, current_y:current_y + patch_size]
            d_patch = d[:, current_x:current_x + patch_size, current_y:current_y + patch_size]

            loss1 = criterion(a_patch, b_patch)
            loss2 = criterion(b_patch, c_patch)
            loss3 = criterion(c_patch, d_patch)
            loss = loss1 + loss2 + loss3;


            if loss >= max_various:
                max_various = loss;
                max_patch_x = current_x
                max_patch_y = current_y
            current_y = current_y+step_size;
        current_x = current_x + step_size

    # a_patch = a[:,max_patch_x:max_patch_x+patch_size,max_patch_y:max_patch_y+patch_size]
    # image = a_patch.detach()
    # image = image.cpu().numpy()
    # image = image.transpose(1, 2, 0)
    #     # image = image * 255
    # plt.imshow(image)
    # plt.show();
    return max_patch_x,max_patch_y;


def getPatchLossMask_stepsize(sample_clip,recon_clip):

    img1=sample_clip[:,0,::]
    img2=sample_clip[:,5,::]
    img3=sample_clip[:,10,::]
    img4=sample_clip[:,15,::]
    imgs = [img1,img2,img3,img4]

    a,b,c,d = imgs

    max_various = 0;
    step_size = int(patch_size/4)
    patch_col = int((112 - patch_size) / step_size) + 1
    patch_loss = torch.zeros((patch_col, patch_col))
    max_patch_x = 0
    max_patch_y = 0
    criterion = nn.MSELoss().cuda()
    current_x = 0;
    x_i = 0
    y_i = 0
    while current_x <= 112-patch_size:
        current_y = 0;
        y_i = 0
        while current_y <= 112-patch_size:
            a_patch = a[:, current_x:current_x + patch_size, current_y:current_y + patch_size]
            b_patch = b[:, current_x:current_x + patch_size, current_y:current_y + patch_size]
            c_patch = c[:, current_x:current_x + patch_size, current_y:current_y + patch_size]
            d_patch = d[:, current_x:current_x + patch_size, current_y:current_y + patch_size]

            loss1 = criterion(a_patch, b_patch)
            loss2 = criterion(b_patch, c_patch)
            loss3 = criterion(c_patch, d_patch)
            loss = loss1 + loss2 + loss3;
            patch_loss[x_i, y_i] = loss

            if loss >= max_various:
                max_various = loss;
                max_patch_x = current_x
                max_patch_y = current_y
            y_i += 1
            current_y = current_y+step_size;
        x_i += 1
        current_x = current_x + step_size

    patch_loss = 1.2 / (patch_loss.max() - patch_loss.min()) * (patch_loss - patch_loss.min()) + 0.8
    c, t, h, w = recon_clip.size()
    patch_loss_mask = torch.ones((c, t, patch_loss.size(0), patch_loss.size(1)))
    patch_loss_mask = patch_loss_mask[:, ] * patch_loss
    patch_loss_mask = F.interpolate(patch_loss_mask, size=(h, w), mode='bilinear', align_corners=False)

    return max_patch_x,max_patch_y,patch_loss_mask




