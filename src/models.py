import torch
import torch.nn as nn
from torch.nn import functional as F

class U_net(nn.Module):
  def __init__(self, width):
    super(U_net, self).__init__()

    # Encoder layers
    self.encoder1 = self.conv_block(1, width)
    self.encoder2 = self.conv_block(width, width*2)

    self.pool = nn.MaxPool2d(2) 

    self.bottleneck = self.conv_block(width*2, width*4)

    # Decoder layers
    self.transpose_conv1 = nn.ConvTranspose2d(width*4, width*2, kernel_size=2, stride=2)
    self.decoder1 = self.conv_block(width*4, width*2)

    self.transpose_conv2 = nn.ConvTranspose2d(width*2, width, kernel_size=2, stride=2)
    self.decoder2 = self.conv_block(width*2, width)

    self.final_layer = nn.Conv2d(width, 1, kernel_size=1)

  def conv_block(self, in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='circular'),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


  def forward(self, x): # (B, 1, 160, 160)
    # Encoding
    enc1 = self.encoder1(x) # (B, 32, 160, 160)
    pool1 = self.pool(enc1) # (B, 32, 80, 80)

    enc2 = self.encoder2(pool1) # (B, 64, 80, 80)
    pool2 = self.pool(enc2) # (B, 64, 40, 40)

    bottleneck = self.bottleneck(pool2) # (B, 128, 40, 40)

    # Decoding
    trans_conv1 = self.transpose_conv1(bottleneck) # (B, 64, 80, 80)
    res1 = torch.cat((trans_conv1, enc2), dim=1) #(B, 64, 80, 80)
    dec1 = self.decoder1(res1) # (B, 64, 80, 80)

    trans_conv2 = self.transpose_conv2(dec1) # (B, 64, 160, 160)
    res2 = torch.cat((trans_conv2, enc1), dim=1) # (B, 64, 160, 160)
    dec2 = self.decoder2(res2) # (B, 32, 160, 160)

    final = self.final_layer(dec2) # (B, 1, 160, 160)

    return final


class SpectralConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        

        scale = (1/(in_channels*out_channels))
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]

        # Fourier transform
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros_like(x_ft)

        out_ft[:, :, :self.modes1, :self.modes2] = self.mat_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.mat_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Inverse transform
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def mat_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    


class FNO(nn.Module):
    def __init__(self, modes1, modes2, width, n_layers = 4):
        super(FNO, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # Lifting
        self.p = nn.Linear(1, self.width)

        # Fourier layers
        self.spectral_convs = nn.ModuleList([SpectralConv2D(width, width, modes1, modes2) for _ in range(n_layers)])
        self.skip_convs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])

        # Projection
        self.q = nn.Linear(self.width, 128)
        self.fc = nn.Linear(128, 1)

    
    def forward(self, x):

        x = x.squeeze(1) 
        x = self.p(x.unsqueeze(-1)) 
        
        x = x.permute(0, 3, 1, 2)

        for s_conv, w_conv in zip(self.spectral_convs, self.skip_convs):
            x = F.gelu(s_conv(x) + w_conv(x))

        x = x.permute(0, 2, 3, 1) # (B, 64, 64, Width)
        x = self.fc(F.gelu(self.q(x))) # (B, 64, 64, 1)

        return x.permute(0, 3, 1, 2) 