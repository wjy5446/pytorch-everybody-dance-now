import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral

class Generator(nn.Module):
    
    def __init__(self, input_nc, output_nc, ngf=64, 
                 n_blocks=9, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        
        assert(n_blocks >= 0)
        super(Generator, self).__init__()
        
        model = [nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                norm_layer(ngf),
                nn.ReLU(True)]
        
        ## downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                     norm_layer(ngf * mult * 2),
                     nn.ReLU(True)]
        
        ## resblock
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer)]
        
        ## upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 
                                         kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                     norm_layer(int(ngf * mult / 2)),
                     nn.ReLU(True)]
        
        model += [nn.ReflectionPad2d(3),
                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                 nn.Tanh()]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        return self.model(input)

class ResnetBlock(nn.Module):
    
    def __init__(self, dim, norm_layer):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer)
    
    def build_conv_block(self, dim, norm_layer):
        conv_block = []
        
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       norm_layer(dim),
                       nn.ReLU(True)]
        
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        output = x + self.conv_block(x)
        return output

class MultiscaleDiscriminator(nn.Module):
    '''
    2차원으로 구성 (num_D, 1)
    '''
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, getIntermFeat=getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)
    
    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]
    
    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            
            result.append(self.singleD_forward(model, input_downsampled))
            
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        
        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, getIntermFeat=False):
        '''
        conv2d, leakyrelu -> (conv2d, norm, leakyrelu) * 3 -> conv2d, norm, leakyrelu -> conv2d
        '''
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        
        sequence = [[spectral(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=2)), 
                     nn.LeakyReLU(0.2, True)]]
        
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                spectral(nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=2)),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]
        
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            spectral(nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=2)),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]
        
        sequence += [[spectral(nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=2))]]
        
        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
        
            self.model = nn.Sequential(*sequence_stream)
        
    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
            