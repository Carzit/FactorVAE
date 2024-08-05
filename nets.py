import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import modules_weight_init

class FeatureExtractor(nn.Module):
    """
    Feature extractor extracts stocks latent features e from the historical sequential characteristics x, formulated as 

        e = φ_feat(x)

    In order to capture the temporal dependence in sequences, we adopt the Gate Recurrent Unit(GRU), a variant of RNN (Chung et al. 2014).
    """
    def __init__(self, input_size, hidden_size, num_layers, drop_out=0.1) -> None:
        super(FeatureExtractor, self).__init__()
        self.norm_layer = nn.LayerNorm(input_size)
        self.proj_layer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LeakyReLU())
        
        self.gru_layer = nn.GRU(hidden_size, 
                                hidden_size, 
                                num_layers, 
                                batch_first=False, 
                                dropout=drop_out)

    def forward(self, x):# input: [seq_len, num_stocks, input_size(num_features)]
        x = self.norm_layer(x)
        x_proj = self.proj_layer(x) # -> x_proj: [seq_len, num_stocks, hidden_size]
        #print("proj_x:",x_proj)
        _, h = self.gru_layer(x_proj) # -> h: [num_layers, batch_size, hidden_size]
        h = h.permute(1, 0, 2) # -> h: [batch_size, num_layers, hidden_size]
        e = h.reshape(h.shape[0], -1) # -> e: [batch_size, num_layers * hidden_size]

        return e

class PortfolioLayer(nn.Module):
    """
    Because the number of individual stocks in cross-section is large and varies with time, 
    instead of using stock returns y directly, we construct a set of portfolios inspired by (Gu, Kelly, and Xiu 2021), 
    these portfolios are dynamically re-weighted on the basis of stock latent features, i.e., 
        y_p = y * φ_p(e) = y * a_p , 
    where a_p ∈ R^M denotes the weight of M portfolios.

        a_p_(i,j) = softmax(w_p * e_i + b_p)
        y_p_(j) = ∑ y(i) * a_p_(i,j)
    where a_p_(i,j) denotes the weight of i-th stock in j-th portfolio and meets ∑a_p_(i,j) = 1, y_p ∈ R^M is the vector of portfolio returns. 
    The main advantages of constructing portfolios lie in: 1) reducing the input dimension and avoiding too many parameters. 2) robust to the missing stocks in cross-section and thus suitable for the market
    """

    def __init__(self, num_portfolios, input_size) -> None:
        super(PortfolioLayer, self).__init__()
        self.w_p = nn.Parameter(torch.randn(num_portfolios, input_size))
        self.b_p = nn.Parameter(torch.zeros(num_portfolios, 1))

    def forward(self, y:torch.Tensor, e:torch.Tensor): # y: [num_stocks] e: [num_stocks, input_size]
        a_p = F.softmax(torch.matmul(self.w_p, e.T) + self.b_p, dim=-1) #-> [num_portfolios, num_stocks]
        y_p = torch.sum(a_p * y, dim=-1) #-> [num_portfolios]
        return y_p


class FactorEncoder(nn.Module):
    """
    Factor encoder extracts posterior factors `z_post` from the future stock returns `y` and the latent features `e`
        [μ_post, σ_post] = φ_enc(y, e)
        z_post ~ N (μ_post, σ_post^2)
    where `z_post` is a random vector following the independent Gaussian distribution, 
    which can be described by the mean `μ_post` ∈ R^K and the standard deviation σ_post ∈ R^K, K is the number of factors.

    Because the number of individual stocks in cross-section is large and varies with time, 
    instead of using stock returns y directly, we construct a set of portfolios inspired by (Gu, Kelly, and Xiu 2021), 
    these portfolios are dynamically re-weighted on the basis of stock latent features, i.e., 
        y_p = y · φ_p(e) = y · a_p , 
    where a_p ∈ R^M denotes the weight of M portfolios.

        a_p_(i,j) = softmax(w_p * e_i + b_p)
        y_p_(j) = ∑ y(i) * a_p_(i,j)
    where a_p_(i,j) denotes the weight of i-th stock in j-th portfolio and meets ∑a_p_(i,j) = 1, y_p ∈ R^M is the vector of portfolio returns. 
    The main advantages of constructing portfolios lie in: 1) reducing the input dimension and avoiding too many parameters. 2) robust to the missing stocks in cross-section and thus suitable for the market

    And then the mean and the std of posterior factors are output by a mapping layer [μ_post, σ_post] = φ_map(y_p)
        μ_post = w * y_p + b
        σ_post = Softplus(w * y_p + b)
    where Softplus(x) = log(1 + exp(x))

    """
    def __init__(self, input_size, hidden_size, latent_size) -> None:
        super(FactorEncoder, self).__init__()
        self.portfoliolayer = PortfolioLayer(input_size=input_size, num_portfolios=hidden_size)
        self.map_mu_z_layer = nn.Linear(hidden_size, latent_size)
        self.map_sigma_z_layer = nn.Linear(hidden_size, latent_size)

    def forward(self, y:torch.Tensor, e:torch.Tensor):# y: [num_stocks] e: [num_stocks, num_features]
        y_p = self.portfoliolayer(y, e) #-> [hidden_size(num_portfolios)]
        mu_z = self.map_mu_z_layer(y_p) #-> [latent_size(num_factors)]
        sigma_z = torch.exp(self.map_sigma_z_layer(y_p)) #-> [latent_size(num_factors)]
        return mu_z, sigma_z

class AlphaLayer(nn.Module):
    """
    Alpha layer outputs idiosyncratic returns α from the latent features e. 
    We assume that α is a Gaussian random vector described by 
        α ~ N (μ_α, σ_α^2)
    where the mean μ_α ∈ R^N and the std σ_α ∈ R^N are output by a distribution network π_α, i.e., [μ_α, σ_α] = π_α(e). 
    Specifically,
        h_α = LeakyReLU(w_α * e + b_α)
        μ_α = w_μ_α * h_α + b_μ_α
        σ_α = Softplus(w_σ_α * h_α + b_σ_α)
    where h_α ∈ R^H is the hidden state.
    """
    def __init__(self, input_size, hidden_size) -> None:
        super(AlphaLayer, self).__init__()
        self.alpha_h_layer = nn.Sequential(nn.Linear(input_size, hidden_size),
                                           nn.LeakyReLU())
        self.alpha_mu_layer = nn.Linear(hidden_size, 1)
        self.alpha_sigma_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, e):#e: [num_stocks, num_features]
        h_alpha = self.alpha_h_layer(e) #->[num_stocks, num_portfolios(hidden_size)]
        mu_alpha = self.alpha_mu_layer(h_alpha).squeeze() #->[num_stocks]
        sigma_alpha = torch.exp(self.alpha_sigma_layer(h_alpha).squeeze()) #->[num_stocks]
        return mu_alpha, sigma_alpha
    
class BetaLayer(nn.Module):
    """
    Beta layer calculates factor exposure β ∈ R^{N*K} from the latent features e by linear mapping. Formally,
        β = φ_β(e) = w_β * e + b_β
    """
    def __init__(self, input_size, latent_size) -> None:
        super(BetaLayer, self).__init__()
        self.beta_layer = nn.Linear(input_size, latent_size)
    
    def forward(self, e):#e: [num_stocks, num_features(input_size)]
        beta = self.beta_layer(e) 
        return beta #->[num_stocks, num_fators(latent_size)]

class FactorDecoder(nn.Module):
    """
    Factor decoder uses factors z and the latent feature e to calculate stock returns `y_hat`
        y_hat = φ_dec(z, e) = α + β * z
    Essentially, the decoder network φdec consists of alpha layer and beta layer.

    Note that α and z are both follow independent Gaussian distribution, and thus the output of decoder y_hat ~ N(μ_y , σ_y^2), where
        μ_y = μ_α + ∑ β_k * μ_z_k
        σ_y = \sqrt{ σ_α^2 + ∑ β_k ^ 2 σ_z_k^2 }
    where μ_z , σ_z ∈ R^K are the mean and the std of factors respectively.
    """
    def __init__(self, input_size, hidden_size, latent_size) -> None: 
        super(FactorDecoder, self).__init__()
        self.alpha_layer = AlphaLayer(input_size, hidden_size)
        self.beta_layer = BetaLayer(input_size, latent_size)
    
    def forward(self, e, mu_z, sigma_z):# e: [num_stocks, num_features], mu_z: [latent_size(num_factors)], sigma_z: [latent_size(num_factors)]
        mu_alpha, sigma_alpha = self.alpha_layer(e) #->[num_stocks]
        beta = self.beta_layer(e) #->[num_stocks, latent_size(num_fators)]

        mu_y = mu_alpha + torch.sum(mu_z * beta, dim=-1) #->[num_stocks]
        sigma_y = torch.sqrt(sigma_alpha ** 2 + torch.sum(beta ** 2 * sigma_z ** 2, dim=-1)) #->[num_stocks]
        y = self.reparameterization(mu_y, sigma_y) #->[num_stocks]
        return y

    def reparameterization(self, mean, std):
        epsilon = torch.randn_like(std).to(next(self.parameters()).device)      
        y = mean + std * epsilon
        return y

class SingleHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SingleHeadAttention, self).__init__()
        self.q = nn.Parameter(torch.randn(hidden_size))
        self.w_key = nn.Linear(input_size, hidden_size, bias=False)
        self.w_value = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, e):# e: [num_stocks, input_size(features)]
        k = self.w_key(e)  # -> (N, H)
        v = self.w_value(e)  # -> (N, H)
        
        q_norm = self.q / self.q.norm(dim=-1, keepdim=True)  # -> (H)
        k_norm = k / k.norm(dim=-1, keepdim=True)  # -> (N, H)
        
        attn_scores = torch.matmul(q_norm, k_norm.transpose(-1,-2))  # (N)
        attn_weights = attn_scores / attn_scores.sum(dim=-1, keepdim=True)  # (N)
        
        h_att = torch.matmul(attn_weights, v)  # (H)
        return h_att

class MultiHeadAttention(nn.Module):
    """
    Considering that a factor usually represents a certain type of risk premium in the market (such as the size factor focuses on the risk premium of small-cap stocks), we design a muti-head global attention mechanism to integrate the diverse global representations of the market in parallel, and extract factors from them to represent diverse risk premium of market. Formally, a single-head attention performs as

        k_i = w_key * e_i, v_i = w_value * e_i
        a_i = relu(q × k_i.T) / norm(q) / norm(k_i)
        h_att = φ_att(e) = ∑ a_i * v_i

    where query token q ∈ R^H is a learnable parameter, and h_att ∈ R^H is the global representation of market. 
    The muti-head attention concatenates K independent heads together 

        h_muti = Concat([φ_att_1(e), . . . , φ_att_K(e)]) 
    
    where h_muti ∈ R^(K * H) is the muti-global representation.
    """
    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([SingleHeadAttention(input_size, hidden_size) for _ in range(num_heads)])
        
    def forward(self, e):
        head_outputs = [head(e) for head in self.heads]
        h_muti = torch.stack(head_outputs, dim=-2)
        return h_muti #->(K, H)
    
class DistributionNetwork(nn.Module):
    """
    And then we use a distribution network πprior to predict the mean µ_prior and the std σ_prior of prior factors z_prior.

        [µ_prior, σ_prior] = π_prior(h_muti)

    """
    def __init__(self, hidden_size):
        super(DistributionNetwork, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, h_multi):#h_multi: [num_factors, hidden_size]
        mu_prior = self.mu_layer(h_multi).squeeze() #->[num_factors]
        sigma_prior = torch.exp(self.sigma_layer(h_multi).squeeze()) #->[num_factors]
        return mu_prior, sigma_prior

class FactorPredictor(nn.Module):
    """
    Factor predictor extracts prior factors z_prior from the stock latent features e:
        [μ_prior, σ_prior] = φ_pred(e)
        z_prior ∼ N (μ_prior, σ_prior^2)
    where z_prior is a Gaussian random vector, described by the mean μ_prior ∈ R^K and the std σprior ∈ R^K, K is the number of factors. 
    """
    def __init__(self, input_size, hidden_size, latent_size) -> None:
        super(FactorPredictor, self).__init__()
        self.multihead_attention = MultiHeadAttention(input_size, hidden_size, latent_size)
        self.distribution_network = DistributionNetwork(hidden_size)

    def forward(self, e):
        h_multi = self.multihead_attention(e)
        mu_prior, sigma_prior = self.distribution_network(h_multi)
        return mu_prior, sigma_prior
    
class FactorVAE(nn.Module):
    """
    Pytorch Implementation of FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns (https://ojs.aaai.org/index.php/AAAI/article/view/20369)

    Our model follows the encoder-decoder architecture of VAE, to learn an optimal factor model, which can reconstruct the cross-sectional stock returns by several factors well. As shown in Figure 3, with access to future stock returns, the encoder plays a role as an oracle, which can extract optimal factors from future data, called posterior factors, and then the decoder reconstructs future stock returns by the posterior factors. Specially, the factors in our model are regarded as the latent variables in VAE, with the capacity of modeling noisy data. 
    Concretely, this architecture contains three components: feature extractor, factor encoder and factor decoder.
    """
    def __init__(self, 
                 input_size, 
                 num_gru_layers, 
                 gru_hidden_size,
                 hidden_size,
                 latent_size,
                 gru_drop_out = 0.1) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_size=input_size, 
                                                  hidden_size=gru_hidden_size, 
                                                  num_layers=num_gru_layers, 
                                                  drop_out=gru_drop_out)
        self.encoder = FactorEncoder(input_size=gru_hidden_size * num_gru_layers, 
                                     hidden_size=hidden_size,
                                     latent_size=latent_size)
        self.predictor = FactorPredictor(input_size=gru_hidden_size * num_gru_layers,
                                         hidden_size=hidden_size,
                                         latent_size=latent_size)
        self.decoder = FactorDecoder(input_size=gru_hidden_size * num_gru_layers,
                                     hidden_size=hidden_size,
                                     latent_size=latent_size)
    def forward(self, x, y):
        e = self.feature_extractor(x)
        mu_posterior, sigma_posterior = self.encoder(y, e)
        mu_prior, sigma_prior = self.predictor(e)
        y_hat = self.decoder(e, mu_posterior, sigma_posterior)
        return y_hat, mu_posterior, sigma_posterior, mu_prior, sigma_prior
    
    def predict(self, x):
        e = self.feature_extractor(x)
        mu_prior, sigma_prior = self.predictor(e)
        y_pred = self.decoder(e, mu_prior, sigma_prior)
        return y_pred, mu_prior, sigma_prior 
    

        


