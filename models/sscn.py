import torch
import torch.nn as nn
from models.c3d import C3D
from models.net_part import *

class SSCN_OneClip(nn.Module):
    def __init__(self,base_network,with_classifier=False,adaptive_contact=False,num_classes=4,with_Tpyramid=False,with_ClsCycle=False):

        super(SSCN_OneClip,self).__init__()

        self.base_network = base_network
        self.with_classifier = with_classifier
        self.adaptive_contact = adaptive_contact
        self.num_classes = num_classes
        self.with_Tpyramid = with_Tpyramid
        self.with_ClsCycle = with_ClsCycle

#         self.conv6 = conv3d(512, 512)
#         self.pool6 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1))

        self.up1 = up3d(512,256)
        self.up2 = up3d(256,128)
        self.up3 = up3d(128,64)

        self.up4_1 = nn.ConvTranspose3d(64,64,kernel_size=(2,2,2),stride=(2,2,2))

        self.outc = conv3d(64,3)

        if self.with_classifier:

            self.conv6_cls = conv3d(512, 512)
            self.conv6_rec = conv3d(512, 512)

            self.pool5 = nn.AdaptiveAvgPool3d(1)
#             self.fc7 = nn.Linear(512*2, 512)
            self.fc8 = nn.Linear(512, self.num_classes)
#             self.relu = nn.ReLU(inplace=True)
#             self.dropout = nn.Dropout(p=0.5)

            if self.with_ClsCycle:
                self.base_network_cyc = base_network
                self.pool5_cyc = nn.AdaptiveAvgPool3d(1)
                self.fc8_cyc = nn.Linear(512, self.num_classes)




    def forward(self, x,tuple_order=None):

        if self.with_Tpyramid:
            f_x = []
            for i in range(x.size(1)):
                x_i = x[:, i, :, :, :, :]
                f_x.append(self.base_network(x_i))
            # fuse f_x1[0] with f_x1[1,2] --> x1
            # fuse f_x2[0] with f_x2[1,2] --> x2
            x = f_x[0]
        else:
            x = self.base_network(x)

        if self.with_classifier:

            x_cls = self.conv6_cls(x)

            h = self.pool5(x_cls)
            h = h.view(-1,h.size(1))
            h = self.fc8(h)

            x = self.conv6_rec(x)


        x = self.up1(x)

        x = self.up2(x)

        x = self.up3(x)
        x = self.up4_1(x)
        x = self.outc(x)



        if self.with_classifier:
            return x,h
        else:
            return x




if __name__ == '__main__':
    base=C3D(with_classifier=False);
    sscn=SSCN_OneClip(base,with_classifier=True);

    input_tensor = torch.autograd.Variable(torch.rand(4, 3, 16, 112, 112))
    #print(input_tensor)
    out,h = sscn(input_tensor,input_tensor)
    # m = nn.ConvTranspose3d(16,33,3,stride=2)
    # input_tensor = torch.autograd.Variable(torch.rand(20,16,10,50,100))
    # out = m(input_tensor)
    print(out.shape)

   # out = out[:,:,16,:,:]

