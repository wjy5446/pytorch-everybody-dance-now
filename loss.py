import torch
import torch.nn as nn
from network import Vgg19

class GANLoss(nn.Module):
    def __init__(self, gan_mode, device, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        self.device = device
        
        self.gan_mode = gan_mode
        self.Tensor = torch.FloatTensor
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
    
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.Tensor(prediction.size()).fill_(self.target_real_label)
        else:
            target_tensor = self.Tensor(prediction.size()).fill_(self.target_fake_label)
        return target_tensor
    
    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan']:
            if isinstance(prediction[0], list):
                loss = 0
                for pred_i in prediction:
                    pred = pred_i[-1]
                    target_tensor = self.get_target_tensor(pred, target_is_real)
                    loss += self.loss(pred, target_tensor.to(self.device))
                return loss
            else:
                target_tensor = self.get_target_tensor(input[-1], target_is_real)
                return self.loss(input[-1], target_tensor.to(self.device))
            
        elif self.gan_mode == 'wgangp':
            if self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
        return loss

class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class FMLoss(nn.Module):
    def __init__(self, device, num_D=3, n_layers=3):
        super(FMLoss, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.criterion = nn.L1Loss()
        self.feat_weight = (4.0 / n_layers + 1)
        self.D_weight = 1.0 / num_D

    def forward(self, fake, real):              
        loss = 0
        for i in range(self.num_D):
            for j in range(len(fake[i])-1):
                loss += self.D_weight * self.feat_weight * \
                self.criterion(fake[i][j], real[i][j].detach())
        return loss