from torch import nn
class conv3d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(conv3d,self).__init__();
        self.conv3d =nn.Sequential(
            nn.Conv3d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),

        )

    def forward(self, x):
        x = self.conv3d(x)
        return x


class down3d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(down3d,self).__init__();
        self.mpcon3d= nn.Sequential(
            conv3d(in_ch,out_ch),
            nn.MaxPool3d(kernel_size=2,stride=2)
        )
    def forward(self, x):
        x = self.mpcon3d(x)
        return x
class up3d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(up3d,self).__init__()

        self.up3d = nn.ConvTranspose3d(in_ch,in_ch,2,stride=2)
        #self.up3d = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv3d = conv3d(in_ch,out_ch);

    def forward(self, x):
        x = self.up3d(x)
        x = self.conv3d(x)

        return x;
class inc(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(inc,self).__init__()
        self.conv = nn.Conv3d(in_ch,out_ch,kernel_size=3,padding=1)

class out(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(out,self).__init__()
        self.conv = nn.Conv3d(in_ch,out_ch,1);
    def forward(self,x):
        x = self.conv(x)

        return x