import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, dilation=1):
        super(depthwise_separable_conv, self).__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), dilation=dilation, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), stride=1, padding=(0, 1), dilation=dilation, groups=in_channels)
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)


    def forward(self, x):
        inputs = x
        x = self.depth_conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.point_conv(x)
        x = self.bn2(x)
        cut = self.shortcut(inputs)
        x = F.relu(x+cut)
        return x
class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, out_ch, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
            # nn.Conv2d(out_ch, out_ch, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
            # nn.Conv2d(out_ch, out_ch, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
            depthwise_separable_conv(in_ch, out_ch),
            depthwise_separable_conv(out_ch, out_ch)
            
        )
        

    def forward(self, x):

        out = self.conv(x)
        return out

class single_conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(single_conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            depthwise_separable_conv(in_ch, out_ch)
        )
        

    def forward(self, x):

        out = self.conv(x)
        return out


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            depthwise_separable_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    

class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()
        
        c_dim_in = dim_in//4
        k_size=3
        pad=(k_size-1) // 2
        
        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=(k_size, 1), padding=(pad, 0), groups=c_dim_in), nn.GELU(),
                                     nn.Conv2d(c_dim_in, c_dim_in, kernel_size=(1, k_size), padding=(0, pad), groups=c_dim_in), nn.GELU(),
                                     nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
                nn.Conv2d(c_dim_in, c_dim_in, 1),
                nn.GELU(),
                nn.Conv2d(c_dim_in, c_dim_in, kernel_size=(3, 1), padding=(1, 0), groups=c_dim_in),
                nn.GELU(),
                nn.Conv2d(c_dim_in, c_dim_in, kernel_size=(1, 3), padding=(0, 1), groups=c_dim_in),
        )
        
        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        
        self.ldw = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), padding=(1, 0), groups=dim_in),
                nn.GELU(),
                nn.Conv2d(dim_in, dim_in, kernel_size=(1, 3), padding=(0, 1), groups=dim_in),
                nn.GELU(),
                nn.Conv2d(dim_in, dim_out, 1),
        )
        
    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        #----------xy----------#
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4],mode='bilinear', align_corners=True))
        #----------zx----------#
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(F.interpolate(params_zx, size=x2.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        #----------zy----------#
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(F.interpolate(params_zy, size=x3.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        #----------dw----------#
        x4 = self.dw(x4)
        #----------concat----------#
        x = torch.cat([x1,x2,x3,x4],dim=1)
        #----------ldw----------#
        x = self.norm2(x)
        x = self.ldw(x)
        return x



class M_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(M_Net, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.average = nn.AvgPool2d(2, 2)
        
        self.upsample1 = nn.Upsample(scale_factor=8)
        self.upsample2 = nn.Upsample(scale_factor=4)
        self.upsample3 = nn.Upsample(scale_factor=2)
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[1], filters[1])
        self.Conv3 = Grouped_multi_axis_Hadamard_Product_Attention(filters[2], filters[2])
        self.Conv4 = Grouped_multi_axis_Hadamard_Product_Attention(filters[3], filters[3])
        self.Conv5 = Grouped_multi_axis_Hadamard_Product_Attention(filters[3], filters[4])
        
        # self.singleConv1 = single_conv_block(in_ch, filters[0])
        self.singleConv2 = single_conv_block(in_ch, filters[0])
        self.singleConv3 = single_conv_block(in_ch, filters[1])
        self.singleConv4 = single_conv_block(in_ch, filters[2])
        self.singleConv5 = single_conv_block(filters[3], filters[4])
        

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = Grouped_multi_axis_Hadamard_Product_Attention(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = Grouped_multi_axis_Hadamard_Product_Attention(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Sequential(
            nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.Conv6 = nn.Sequential(
            nn.Conv2d(filters[3], out_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.Conv7 = nn.Sequential(
            nn.Conv2d(filters[2], out_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.Conv8 = nn.Sequential(
            nn.Conv2d(filters[1], out_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.final = nn.Conv2d(out_ch*4, out_ch, 1, 1, bias=False)
       # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        
        # leg
        scale_img2 = self.average(x)
        scale_img3 = self.average(scale_img2)
        scale_img4 = self.average(scale_img3)
#         print('sc', scale_img2.shape)
        
        
        # print(scale_img2.shape)
        
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)
        # print('e2', e2.shape)
        
        input2 = self.singleConv2(scale_img2)
#         print('input2', input2.shape)
        input2 = torch.cat([input2, e2], dim=1)
        
        
        e2 = self.Conv2(input2)
        e3 = self.Maxpool2(e2)
        
        input3 = self.singleConv3(scale_img3)
        # print(input3.shape)
        input3 = torch.cat([input3, e3], dim=1)
        # print(input3.shape)
        
        e3 = self.Conv3(input3)
        e4 = self.Maxpool3(e3)
        
        input4 = self.singleConv4(scale_img4)
        # print(input3.shape)
        input4 = torch.cat([input4, e4], dim=1)
        
        e4 = self.Conv4(input4)
        e5 = self.Maxpool4(e4)
        
        e5 = self.Conv5(e5)

        # upsample block
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        side6 = self.upsample1(d5)
        side7 = self.upsample2(d4)
        side8 = self.upsample3(d3)
        
        out6 = self.Conv6(side6)
        out7 = self.Conv7(side7)
        out8 = self.Conv8(side8)
        out9 = self.Conv(d2)
        
        out = torch.cat([out6, out7, out8, out9], dim=1)
        out = self.final(out)
        # out = d5

        #d1 = self.active(out)

        return out
    
if __name__ == '__main__':
    import time
    start_time = time.time()
    model = M_Net(2, 1)
    print(model(torch.randn(2, 2, 256, 256)).shape)
    time_all = time.time() - start_time 
    print('test finished!! and Total time %.4f seconds for training' % (time_all))
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))