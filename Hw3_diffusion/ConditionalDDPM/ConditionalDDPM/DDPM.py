import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
        beta = torch.linspace(beta_1, beta_T, T).to(device)
        beta_T = beta[t_s-1]
        sqrt_beta_t = torch.sqrt(beta_T)
        alpha_t = 1-beta_T
        oneover_sqrt_alpha = 1/torch.sqrt(alpha_t)
        alpha_t_bar = torch.cumprod(1-beta, dim=0)[t_s-1]
        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
        sqrt_oneminus_alpha_bar = torch.sqrt(1-alpha_t_bar)
        # ==================================================== #
        return {
            'beta_t': beta_T,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    def forward(self, images, conditions):
        T = self.dmconfig.T
        noise_loss = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function  .  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        B = images.shape[0]
        t_s = torch.randint(1,T+1, size=(B,1)).to(device)
        t_s_norm = (t_s-1.0)/(T-1)
        noise = torch.randn_like(images).to(device)
        conditions = F.one_hot(conditions, num_classes=self.dmconfig.num_classes).to(device)
        drop_batch_idx = torch.rand(B).to(device) <= self.dmconfig.mask_p
        conditions[drop_batch_idx, :] = self.dmconfig.condition_mask_value
        
        scheduler = self.scheduler(t_s)
        sqrt_alpha_bar = scheduler['sqrt_alpha_bar'].reshape(-1,1,1,1)
        sqrt_oneminus_alpha_bar = scheduler['sqrt_oneminus_alpha_bar'].reshape(-1,1,1,1)
        x_t = sqrt_alpha_bar * images + sqrt_oneminus_alpha_bar * noise

        noise_loss = self.loss_fn(self.network(x_t, t_s_norm.reshape(-1,1,1,1), conditions), noise)
        
        # ==================================================== #
        return noise_loss




    def sample(self, conditions, omega):
        T = self.dmconfig.T
        X_t = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        B = conditions.size(dim=0)
        num_channels = self.dmconfig.num_channels
        input_dim = self.dmconfig.input_dim

        X_t = torch.randn(B, num_channels, input_dim[0], input_dim[1]).to(device)
        my_condition = torch.full_like(conditions, self.dmconfig.condition_mask_value).to(device)

        with torch.no_grad():
            for t_s in torch.arange(T, 0, -1):
                time_step = torch.full((B,1), t_s).to(device)
                time_step_norm = (time_step - 1.0)/ (T-1)

                if t_s > 1:
                    z = torch.randn_like(X_t).to(device)
                else:
                    z = torch.zeros_like(X_t).to(device)

                cond_eps = self.network(X_t, time_step_norm.reshape(-1,1,1,1), conditions)
                uncond_eps = self.network(X_t, time_step_norm.reshape(-1,1,1,1), my_condition)
                eps_t = (1+omega)*cond_eps - omega*uncond_eps

                scheduler = self.scheduler(time_step)
                oneover_sqrt_alpha = scheduler['oneover_sqrt_alpha'].reshape(-1,1,1,1)
                sqrt_oneminus_alpha_bar = scheduler['sqrt_oneminus_alpha_bar'].reshape(-1,1,1,1)
                alpha_t = scheduler['alpha_t'].reshape(-1,1,1,1)
                sqrt_beta_t = scheduler['sqrt_beta_t'].reshape(-1,1,1,1)
                X_t = oneover_sqrt_alpha * (X_t-(1-alpha_t)/sqrt_oneminus_alpha_bar*eps_t) + sqrt_beta_t*z

        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
        return generated_images