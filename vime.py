"""Implementation of VIME: Variational Information Maximizing Exploration.
https://arxiv.org/abs/1605.09674
"""
from collections import deque
import numpy as np
import torch
from torch import nn 
from torch.optim import Adam
from torch.nn import functional as F
from torch.distributions.normal import Normal


def _elements(t: torch.Tensor):
    return np.prod(t.shape)


class _BayesianLinerLayer(nn.Module):
    """A linear layer which samples network parameters on forward calculation. 
    Local re-parameterization trick is used instead of direct sampling of network parameters.
    """
    def __init__(self, fan_in: int, fan_out: int):
        super().__init__()
        self._fan_in, self._fan_out = fan_in, fan_out

        self._W_mu = torch.normal(torch.zeros(fan_in, fan_out), torch.ones(fan_in, fan_out)) # N(0, 1)
        self._W_rho = torch.log(torch.exp(torch.ones(fan_in, fan_out) * 0.5) - 1.) # log(e^0.5 - 1) to make \sigma_0 = 0.5
        self._b_mu = torch.normal(torch.zeros(fan_out), torch.ones(fan_out)) # N(0, 1)
        self._b_rho = torch.log(np.exp(torch.ones(fan_out) * .5) - 1.) # log(e^0.5 - 1) to make \sigma_0 = 0.5

        self._W_var, self._b_var = self._rho2var(self._W_rho), self._rho2var(self._b_rho)
        self._parameter_number = _elements(self._W_mu) + _elements(self._b_mu)
        self._distributional_parameter_number = _elements(self._W_mu) + _elements(self._W_rho) + _elements(self._b_mu) + _elements(self._b_rho)

    @staticmethod
    def _rho2var(rho):
        return torch.log(1. + torch.exp(rho)).pow(2)
        
    @property
    def parameter_number(self):
        return self._parameter_number

    @property
    def distributional_parameter_number(self):
        return self._distributional_parameter_number

    def get_parameters(self):
        """Return all parameters in this layer as vectors of mu and rho.
        """
        params_mu = torch.cat([self._W_mu.data.reshape(-1), self._b_mu.data.reshape(-1)])
        params_rho = torch.cat([self._W_rho.data.reshape(-1), self._b_rho.data.reshape(-1)])
        return params_mu, params_rho

    def set_parameters(self, params_mu: torch.Tensor, params_rho: torch.Tensor):
        """Receive parameters (mu and rho) as vectors and set them.
        """
        assert params_mu.size() == torch.Size([self._parameter_number])
        assert params_rho.size() == torch.Size([self._parameter_number])

        self._W_mu = params_mu[: _elements(self._W_mu)].reshape(self._W_mu.size())
        self._b_mu = params_mu[_elements(self._W_mu) :].reshape(self._b_mu.size())

        self._W_rho = params_rho[: _elements(self._W_rho)].reshape(self._W_rho.size())
        self._b_rho = params_rho[_elements(self._W_rho) :].reshape(self._b_rho.size())

        self._W_var, self._b_var = self._rho2var(self._W_rho), self._rho2var(self._b_rho)

    def forward(self, X, share_paremeters_among_samples=True):
        """Linear forward calculation with local re-parameterization trick.
        params
        ---
        X: (batch, input_size)
        share_paremeters_among_samples: (bool) Use the same set of parameters for samples in a batch
        return
        ---
        r: (batch, output_size)
        """
        gamma = X @ self._W_mu + self._b_mu
        delta = X.pow(2) @ self._W_var + self._b_var

        if share_paremeters_among_samples:
            zeta = Normal(torch.zeros(1, self._fan_out), torch.ones(1, self._fan_out)).sample().repeat([X.size(0), 1])
        else:
            zeta = Normal(torch.zeros(X.size(0), self._fan_out), torch.ones(X.size(0), self._fan_out)).sample()
        zeta = zeta.to(X.device)
        r = gamma + delta.pow(0.5) * zeta
        return r


class BNN:
    def __init__(self, input_size, output_size, hidden_layers, hidden_layer_size, max_logvar, min_logvar):
        self._input_size = input_size
        self._output_size = output_size
        self._max_logvar = max_logvar
        self._min_logvar = min_logvar

        self._hidden_layers = []
        fan_in = self._input_size
        self._parameter_number = 0
        for _ in range(hidden_layers):
            l = _BayesianLinerLayer(fan_in, hidden_layer_size)
            self._hidden_layers.append(l)   
            self._parameter_number += l.parameter_number
            fan_in = hidden_layer_size
        self._out_layer = _BayesianLinerLayer(fan_in, output_size * 2)
        self._parameter_number += self._out_layer.parameter_number
        self._distributional_parameter_number = self._parameter_number * 2

    @property
    def network_parameter_number(self):
        """The number elements in theta."""
        return self._parameter_number

    @property
    def distributional_parameter_number(self):
        """The number elements in phi."""
        return self._distributional_parameter_number

    def get_parameters(self):
        """Return mu and rho as a tuple of vectors. 
        """
        params_mu, params_rho = zip(*[l.get_parameters() for l in self._hidden_layers + [self._out_layer]])
        return torch.cat(params_mu), torch.cat(params_rho)
        
    def set_params(self, params_mu, params_rho):
        """Set a vector of parameters into weights and biases.
        """
        assert params_mu.size() == torch.Size([self._parameter_number]), "expected a vector of {}, got {}".format(self._parameter_number, params_mu.size())
        assert params_rho.size() == torch.Size([self._parameter_number]), "expected a vector of {}, got {}".format(self._parameter_number, params_rho.size())

        begin = 0
        for l in self._hidden_layers + [self._out_layer]:
            end = begin + l.parameter_number
            l.set_parameters(params_mu[begin : end], params_rho[begin : end])
            begin = end

    def infer(self, X, share_paremeters_among_samples=True):
        for layer in self._hidden_layers:
            X = F.relu(layer(X, share_paremeters_among_samples))
        X = self._out_layer(X, share_paremeters_among_samples)
        mean, logvar = X[:, :self._output_size], X[:, self._output_size:]
        logvar = torch.clamp(logvar, min=self._min_logvar, max=self._max_logvar)
        return mean, logvar

    def log_likelihood(self, input_batch, output_batch):
        """Calculate an expectation of log likelihood.
        Mote Carlo approximation using a single parameter sample,
        i.e., E_{theta ~ q(* | phi)} [ log p(D | theta)] ~ log p(D | theta_1)
        """
        output_mean, output_logvar = self.infer(input_batch, share_paremeters_among_samples=True)

        # log p(s_next)
        # = log N(output_batch | output_mean, exp(output_logvar))
        # = -\frac{1}{2} \sum^d_j [ logvar_j + (s_next_j - output_mean)^2 exp(- logvar_j) ]  - \frac{d}{2} \log (2\pi)
        ll = - .5 * ( output_logvar + (output_batch - output_mean).pow(2) * (- output_logvar).exp() ).sum(dim=1) - .5 * self._output_size * np.log(2 * np.pi)
        return ll.mean()

class VIME(nn.Module):
    _ATTRIBUTES_TO_SAVE = [
        '_D_KL_smooth_length', '_prev_D_KL_medians',
        '_eta', '_lamb',
        '_dynamics_model',
        '_params_mu', '_params_rho',
        '_H',
        '_optim',
    ]

    def __init__(self, observation_size, action_size, device='cpu', eta=0.1, lamb=0.01, update_iterations=500, learning_rate=0.0001, 
            hidden_layers=2, hidden_layer_size=64, D_KL_smooth_length=10, max_logvar=2., min_logvar=-10.):
        super().__init__()

        self._update_iterations = update_iterations
        self._eta = eta
        self._lamb = lamb
        self._device = device

        self._D_KL_smooth_length = D_KL_smooth_length
        self._prev_D_KL_medians = deque(maxlen=D_KL_smooth_length)

        self._dynamics_model = BNN(observation_size + action_size, observation_size, hidden_layers, hidden_layer_size, max_logvar, min_logvar)
        init_params_mu, init_params_rho = self._dynamics_model.get_parameters()
        self._params_mu = nn.Parameter(init_params_mu.to(device))
        self._params_rho = nn.Parameter(init_params_rho.to(device))
        self._H = self._calc_hessian()
        self._dynamics_model.set_params(self._params_mu, self._params_rho)
        self._optim = Adam([self._params_mu, self._params_rho], lr=learning_rate) 

    def calc_curiosity_reward(self, rewards: np.ndarray, info_gains: np.ndarray):
        if len(self._prev_D_KL_medians) == 0:
            relative_gains = info_gains
        else:
            relative_gains = info_gains / np.mean(self._prev_D_KL_medians)
        return rewards + self._eta * relative_gains

    def memorize_episodic_info_gains(self, info_gains: np.array):
        """Call this method after collecting a trajectory to save a median of infomation gains throughout the episode.
        Params
        ---
        info_gains: array of D_KLs throughout an episode
        """
        self._prev_D_KL_medians.append(np.median(info_gains))

    def calc_info_gain(self, s, a, s_next):
        """Calculate information gain D_KL[ q( /cdot | \phi') || q( /cdot | \phi_n) ].
        Return info_gain, log-likelihood of each sample \log p(s_{t+1}, a_t, s_)
        """
        self._dynamics_model.set_params(self._params_mu, self._params_rho) # necessary to calculate new gradient
        ll = self._dynamics_model.log_likelihood(
            torch.tensor(np.concatenate([s, a]), dtype=torch.float32).unsqueeze(0).to(self._device),
            torch.tensor(s_next, dtype=torch.float32).unsqueeze(0).to(self._device))
        l = - ll

        self._optim.zero_grad()
        l.backward()  # Calculate gradient \nabla_\phi l ( = \nalba_\phi -E_{\theta \sim q(\cdot | \phi)}[ \log p(s_{t+1} | \s_t, a_t, \theta) ] )
        nabla = torch.cat([self._params_mu.grad.data, self._params_rho.grad.data])

        # \frac{\lambda^2}{2} (\nabla_\phi l)^{\rm T} H^{-1} (\nabla_\phi^{\rm T} l)
        with torch.no_grad():
            info_gain = .5 * self._lamb ** 2 * torch.sum(nabla.pow(2) * self._H.pow(-1))
        return info_gain.cpu().item(), ll.mean().detach().cpu().item()

    def _calc_hessian(self):
        """Return diagonal elements of H = [ \frac{\partial^2 l_{D_{KL}}}{{\partial \phi_j}^2} ]_j
        \frac{\partial^2 l_{D_{KL}}}{{\partial \mu_j}^2} = - \frac{1}{\log^2 (1 + e^{\phi_j})}
        \frac{\partial^2 l_{D_{KL}}}{{\partial \rho_j}^2} = - \frac{1}{\log^2 (1 + e^{\phi_j})} \frac{2 e^{2 \rho_j}}{(1 + e^{rho_j})^2}
        """
        with torch.no_grad():
            denomi = 1 + self._params_rho.exp()
            log_denomi = denomi.log()
            H_mu = log_denomi.pow(-2)
            H_rho = 2 * torch.exp(2 * self._params_rho) / (denomi * log_denomi).pow(2)
            H = torch.cat([H_mu, H_rho])
        return H

    def _calc_div_kl(self):
        """Calculate D_{KL} [ q(\theta | \phi) || p(\theta) ]
        = \frac{1}{2} \sum^d_i [ \log(var^{init}_i) - \log(var_i) + \frac{var_i}{var^{init}_i} + \frac{(\mu_i - \mu^{init}_i)^2}{var^{init}_i} ] - \frac{d}{2}
        """
        var = (1 + self._params_rho.exp()).log().pow(2)
        init_var = torch.ones_like(self._params_rho) * 0.5**2
        return .5 * ( init_var.log() - var.log() + var / init_var + (self._params_mu).pow(2) / init_var ).sum() - .5 * len(self._params_mu)

    def update_posterior(self, batch_s, batch_a, batch_s_next):
        """
        Params
        ---
        batch_s: (batch, observation_size)
        batch_a: (batch, action_size)
        batch_s_next: (batch, observation_size)
        Return
        ---
        loss: (float)
        """
        batch_s = torch.FloatTensor(batch_s).to(self._device)
        batch_a = torch.FloatTensor(batch_a).to(self._device)
        batch_s_next = torch.FloatTensor(batch_s_next).to(self._device)

        self._dynamics_model.set_params(self._params_mu, self._params_rho)
        log_likelihood = self._dynamics_model.log_likelihood(torch.cat([batch_s, batch_a], dim=1), batch_s_next)
        div_kl = self._calc_div_kl()

        elbo = log_likelihood - div_kl
        assert not torch.isnan(elbo).any() and not torch.isinf(elbo).any(), elbo.item()

        self._optim.zero_grad()
        (-elbo).backward()
        self._optim.step()

        # Update hessian
        self._H = self._calc_hessian()

        # Check parameters
        assert not torch.isnan(self._params_mu).any() and not torch.isinf(self._params_mu).any(), self._params_mu
        assert not torch.isnan(self._params_rho).any() and not torch.isinf(self._params_rho).any(), self._params_rho

        # update self._params
        self._dynamics_model.set_params(self._params_mu, self._params_rho)

        return elbo.item()

    def state_dict(self):
        return {
            k: getattr(self, k).state_dict() if hasattr(getattr(self, k), 'state_dict') else getattr(self, k)
            for k in self._ATTRIBUTES_TO_SAVE
        }

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if hasattr(getattr(self, k), 'load_state_dict'):
                getattr(self, k).load_state_dict(v)
            else:
                setattr(self, k, v)  