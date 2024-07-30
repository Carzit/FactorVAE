import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectiveLoss(nn.Module):
    """
    Our objective consists of two parts, the first part is to train an optimal posterior factor model, and the second part is to effectively guide the leaning of factor predictor by the posterior factors.
    Thus, the loss function of model is

        L(x, y) = −(1/N) · ∑ log(P_dec(y_hat_i = y_i|x, z_post)) + γ · KL(P_enc(z|x, y), P_pred(z|x))
    
    where the first loss term is the negative log likelihood, to reduce the reconstruction error of posterior factor model, and y_hat_i = α_i + β_i · z_post is the reconstructed return of i-th stock.The second loss term is the Kullback–Leibler divergence (KLD) between the distribution of prior and posterior factors, for enforcing the prior factors to approximate to the posterior factors, γ is the weight of KLD loss.
    """
    def __init__(self, gamma=1) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, 
                y, 
                y_hat, 
                mu_prior, 
                sigma_prior, 
                mu_posterior, 
                sigma_posterior):

        # Reconstruction loss
        recon_loss = F.mse_loss(y_hat, y, reduction="mean")
        
        # KLD loss
        # kl_divergence between two gaussian distribution:
        # KL(N(x;μ_1, σ_1)||N(x;μ_2, σ_2)) = log（σ_2/σ_1) + (σ_1^2 + (μ_1 - μ_2)^2) / (2·σ_2^2) - 1/2
        kld_loss = torch.sum(torch.log(sigma_prior / sigma_posterior) + (sigma_posterior**2 + (mu_posterior - mu_prior)**2) / (2 * sigma_prior**2) - 0.5)
        
        return recon_loss + self.gamma * kld_loss       

