from torch import nn
import torch

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):  return getattr(nn, activation_type)()
    else:  return nn.ReLU()


class ConvBatchNorm(nn.Module):
    """This block implements the sequence: (convolution => [BN] => ReLU)"""  
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()

        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)
      
    def forward(self, x):
        out_conv = self.conv(x)
        out_norm = self.norm(out_conv)
        out_activation = self.activation(out_norm)
        return out_activation

    
def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv-1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class DownBlock(nn.Module):
    """Downscaling with maxpooling and convolutions"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2,padding=0)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out_maxpool = self.maxpool(x)
        out_nconvs = self.nConvs(out_maxpool)
        return out_nconvs


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv=2, activation='ReLU'):
        super(Bottleneck, self).__init__()
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, input):
        out = self.nConvs(input)
        return out


class UpBlock(nn.Module):
    """Upscaling then conv"""
    def __init__(self, in_channels, out_channels, nb_Conv=2, activation='ReLU'):
        super(UpBlock, self).__init__()      

        self.up = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):

        out_up = self.up(x)
        x = torch.cat((out_up,skip_x),dim=1)
        return self.nConvs(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=3, channels_list=[64,128,256,512,1024], use_pos_dose_mask=False):
        '''
        n_channels : number of channels of the input. 
                        By default 12
        n_labels : number of channels of the ouput.
                      By default 1
        channels_list : list of the number of channels for each block. Must be of length 5.
        use_pos_dose_mask : if True, the model will include a post processing step to remove the dose outside the mask.
        '''
        super(UNet, self).__init__()
        
        assert len(channels_list) == 5, f"channels_list must be of length 5, got {len(channels_list)}"

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_pos_dose_mask = use_pos_dose_mask

        self.inc = ConvBatchNorm(n_channels, channels_list[0])
        self.down1 =  DownBlock(channels_list[0],channels_list[1],2)
        self.down2 =  DownBlock(channels_list[1],channels_list[2],2)
        self.down3 =  DownBlock(channels_list[2],channels_list[3],2)
        self.down4 =  DownBlock(channels_list[3],channels_list[4],2)
        self.Encoder = [self.down1, self.down2, self.down3, self.down4]

        self.bottleneck = Bottleneck(channels_list[4], channels_list[4])

        self.up1 = UpBlock(channels_list[4],channels_list[3],2)
        self.up2 = UpBlock(channels_list[3],channels_list[2],2)
        self.up3 = UpBlock(channels_list[2],channels_list[1],2)
        self.up4 = UpBlock(channels_list[1],channels_list[0],2)
        self.Decoder = [self.up1, self.up2, self.up3, self.up4]

        self.outc = nn.Conv2d(channels_list[0], self.n_classes, kernel_size=3, stride=1, padding=1)

    
    def forward(self, x):
        '''x shape : (batch_size, n_channels, height, width)'''
        # Forward 
        x = self.inc(x)
        skip_inputs = [x] 


        # Forward through encoder
        for i, block in enumerate(self.Encoder):
            x = block(x)  
            skip_inputs += [x] 

        bottleneck = self.bottleneck(x)

        # Forward through decoder
        skip_inputs.reverse()

        decoded = bottleneck
        for i, block in enumerate(self.Decoder):
            # Concat with skipconnections
            skipped = skip_inputs[i+1] # No skip connection for the last block of the encoder
            decoded = block(decoded, skipped)

        out = self.outc(decoded)
        out = torch.softmax(out, dim=1)

        return out


