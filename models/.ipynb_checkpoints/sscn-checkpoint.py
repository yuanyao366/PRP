import torch
import torch.nn as nn
from models.c3d import C3D
from models.net_part import *

class SSCN(nn.Module):
    def __init__(self,base_network):

        super(SSCN,self).__init__()

        self.base_network = base_network;

        self.up1 = up3d(512,256);
        self.up2 = up3d(256,128)
        self.up3 = up3d(128,64)

        self.up4_1 =   nn.ConvTranspose3d(64,64,kernel_size=(1,2,2),stride=(1,2,2))


        self.outc = conv3d(64,3)


    def forward(self, x1,x2):

        x=self.base_network(x1,x2);

        x = self.up1(x)

        x = self.up2(x)

        x = self.up3(x)
        x = self.up4_1(x)
        x = self.outc(x)

   #


        return x;



if __name__ == '__main__':
    base=C3D(with_classifier=False);
    sscn=SSCN(base);

    input_tensor = torch.autograd.Variable(torch.rand(1, 3, 16, 112, 112))
    #print(input_tensor)
    out = sscn(input_tensor,input_tensor)
    # m = nn.ConvTranspose3d(16,33,3,stride=2)
    # input_tensor = torch.autograd.Variable(torch.rand(20,16,10,50,100))
    # out = m(input_tensor)
    print(out.shape)

   # out = out[:,:,16,:,:]
