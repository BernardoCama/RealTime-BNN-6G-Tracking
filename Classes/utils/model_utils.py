import os
import sys
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm

# Directories
cwd = os.path.split(os.path.abspath(__file__))[0]
DB_DIR =  os.path.join(os.path.split(os.path.split(cwd)[0])[0], 'DB')
CLASSES_DIR = os.path.join(cwd, 'Classes')
EXPERIMENTS_DIR = os.path.join(cwd, 'Exp')
PARAMETERS_DIR = os.path.join(cwd, 'Parameters')
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(EXPERIMENTS_DIR))
sys.path.append(os.path.dirname(PARAMETERS_DIR))
sys.path.append(os.path.dirname(cwd))



###########################################################################################
###########################################################################################
# LOSSES

###########################################################################################
# Artificial dataset

# target_log_sigma2_T is the aleatoric uncertainty predicted by the teacher which is used as reference
def log_gaussian_loss(output_y, target_t, output_log_sigma2, target_log_sigma2_T = None):

    if target_log_sigma2_T == None:
        sigma2_T = torch.tensor(0)
    else:
        sigma2_T = target_log_sigma2_T.exp()

    exponent = +0.5*torch.exp(-output_log_sigma2)*((target_t - output_y)**2 + sigma2_T)
    log_coeff = +0.5*output_log_sigma2

    return (log_coeff + exponent).sum()

# target_log_sigma2_T is the epistemic uncertainty predicted by the teacher which is used as reference
def SSE(output_log_sigma2, target_log_sigma2_T):
    return ((output_log_sigma2-target_log_sigma2_T)**2).sum()
def MSE(output_log_sigma2, target_log_sigma2_T):
    return ((output_log_sigma2-target_log_sigma2_T)**2).mean()


###########################################################################################
# Artificial 2D dataset

def log_multivariate_gaussian_loss(params, output_y, target_t, output_sigma2 = None, target_sigma2_T = None):
    
    target_t = target_t.float()
    batch_size = output_y.shape[0]
    output_size = output_y.shape[1]
    epsilon=1e-6

    if output_sigma2 == None:
        output_sigma2 = torch.eye(output_size).repeat(batch_size, 1, 1).to(output_y.device)

    if target_sigma2_T == None:
        regularized_target_covariance_ep = torch.zeros(output_size).repeat(batch_size, 1, 1).to(output_y.device)
    else:
        # Reshape and then transpose to get it to shape [batch_size, output_size, output_size]
        target_covariance_ep = target_sigma2_T.reshape(-1, output_size, output_size).permute(0, 2, 1)
        regularized_target_covariance_ep = target_covariance_ep + epsilon * torch.eye(output_size).to(target_covariance_ep.device).expand_as(target_covariance_ep)
        regularized_target_covariance_ep = 0.5 * (regularized_target_covariance_ep + regularized_target_covariance_ep.transpose(-1, -2))

    output_covariance_al = output_sigma2.reshape(-1, output_size, output_size).permute(0, 2, 1)
    regularized_output_covariance_al = output_covariance_al + epsilon * torch.eye(output_size).to(output_sigma2.device).expand_as(output_covariance_al)
    # Make it symmetric (covariance matrices are symmetric)
    regularized_output_covariance_al = 0.5 * (regularized_output_covariance_al + regularized_output_covariance_al.transpose(-1, -2))


    inv_output_covariance_al = torch.inverse(regularized_output_covariance_al)
    
    first_matrix_mul = torch.bmm(torch.transpose((target_t - output_y).view(batch_size, output_size, -1), 1, 2), inv_output_covariance_al)
    second_matrix_mul = torch.bmm(first_matrix_mul, (target_t - output_y).view(batch_size, output_size, -1)).squeeze()

    if target_sigma2_T != None:
        #                                   +  trace(Cov_T*Cov_S^-1)
        exponent = +0.5*(second_matrix_mul) + torch.sum(torch.diagonal(torch.bmm(regularized_target_covariance_ep, inv_output_covariance_al), dim1=-2, dim2=-1), dim=-1)
    else:
        exponent = +0.5*(second_matrix_mul)
        
    # Check if matrix is positive definite and attempt Cholesky decomposition
    try:
        # First method
        if params.OS == 'Darwin':
            cholesky_result = torch.linalg.cholesky(regularized_output_covariance_al)
        elif params.OS == 'Linux':
            cholesky_result = torch.cholesky(regularized_output_covariance_al)
        log_coeff = torch.sum(torch.log(torch.diagonal(cholesky_result, dim1=-2, dim2=-1)), dim=-1)
        log_coeff = log_coeff * 2  # Because the determinant of the covariance matrix is the square of the determinant of the Cholesky decomposition
        log_coeff = 0.5 * log_coeff

        # Second method
        # log_coeff = 0.5 * torch.logdet(regularized_output_covariance_al)
    except:
        # Add penalty term to induce Positive-Semi definite matrix
        print('Sigma_S is not positive semi-definite\n')
        log_coeff = -torch.det(regularized_output_covariance_al)
        
    return (log_coeff + exponent).mean()


###########################################################################################
# Ray tracing dataset

def product_of_Gaussians(mu, C):

    dim = C.shape[-1]
    mu = np.array(mu).squeeze()
    if len(mu.shape) == 1:
        mu = np.expand_dims(mu, axis=0)

    if mu.shape[1] != dim:
        mu = mu.transpose(1, 0)

    # Initialize sum of inverse covariances and weighted means
    sum_inv_cov = np.zeros((dim, dim))
    sum_weighted_means = np.zeros(dim)
    
    for i in range(len(mu)):
        # Make it symmetric
        C[i] = 0.5 * (C[i] + C[i].transpose(-1, -2))
        inv_cov = np.linalg.inv(C[i])
        sum_inv_cov += inv_cov
        sum_weighted_means += np.dot(inv_cov, mu[i])
    
    # Compute new covariance matrix (C_new)
    C_new = np.linalg.inv(sum_inv_cov)

    # Compute new mean (mu_new)
    mu_new = np.dot(C_new, sum_weighted_means)

    if len(mu_new.shape) == 1:
        mu_new = mu_new.reshape(dim, -1)
    
    return mu_new, C_new


# Modify the function to include the projection step for making the covariance matrix positive semi-definite
def project_to_positive_semidefinite(matrix):
    """Project a square matrix to the closest positive semi-definite matrix."""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvalues[eigenvalues < 0] = 0  # Set negative eigenvalues to zero
    return eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)



###########################################################################################
###########################################################################################
# MODELS


###########################################################################################
# BBP

# Gaussian model with parameters mu and sigma
class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def loglik(self, weights):
        exponent = -0.5*(weights - self.mu)**2/self.sigma**2
        log_coeff = -0.5*(np.log(2*np.pi) + 2*np.log(self.sigma))
        
        return (exponent + log_coeff).sum()

class BayesLinear_Normalq(nn.Module):
    def __init__(self, input_dim, output_dim, prior):
        super(BayesLinear_Normalq, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior = prior
        
        scale = (2/self.input_dim)**0.5
        rho_init = np.log(np.exp((2/self.input_dim)**0.5) - 1)
        self.weight_mus = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        self.weight_rhos = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-3, -3))
        
        self.bias_mus = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))
        self.bias_rhos = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-4, -3))
        
    def forward(self, x, sample = True):
        
        if sample:
            # sample gaussian noise (epsilon) for each weight and each bias
            weight_epsilons = Variable(self.weight_mus.data.new(self.weight_mus.size()).normal_())
            bias_epsilons =  Variable(self.bias_mus.data.new(self.bias_mus.size()).normal_())
            
            # calculate the weight and bias stds from the rho parameters
            weight_stds = torch.log(1 + torch.exp(self.weight_rhos))
            bias_stds = torch.log(1 + torch.exp(self.bias_rhos))
            
            # calculate samples from the posterior from the sampled noise and mus/stds
            weight_sample = self.weight_mus + weight_epsilons*weight_stds
            bias_sample = self.bias_mus + bias_epsilons*bias_stds
            
            # theta (BNN parameters) are the multiplied for the input x
            # DEPENDENT ON THE LAYER TYPE!
            output = torch.mm(x, weight_sample) + bias_sample
            
            # computing the KL loss term = KL ( q_phi(theta)|| P(theta) )
            # P(theta) = self.prior
            # q_phi(theta) = 
            #           N(theta weights | self.weight_mus , weight_stds) AND
            #           N(theta biases | self.bias_mus , bias_stds)
            prior_cov, varpost_cov = self.prior.sigma**2, weight_stds**2
            KL_loss = 0.5*(torch.log(prior_cov/varpost_cov)).sum() - 0.5*weight_stds.numel()
            KL_loss = KL_loss + 0.5*(varpost_cov/prior_cov).sum()
            KL_loss = KL_loss + 0.5*((self.weight_mus - self.prior.mu)**2/prior_cov).sum()
            
            prior_cov, varpost_cov = self.prior.sigma**2, bias_stds**2
            KL_loss = KL_loss + 0.5*(torch.log(prior_cov/varpost_cov)).sum() - 0.5*bias_stds.numel()
            KL_loss = KL_loss + 0.5*(varpost_cov/prior_cov).sum()
            KL_loss = KL_loss + 0.5*((self.bias_mus - self.prior.mu)**2/prior_cov).sum()
            
            return output, KL_loss
        
        else:
            output = torch.mm(x, self.weight_mus) + self.bias_mus
            return output, KL_loss
        
    def sample_layer(self, no_samples):
        all_samples = []
        for i in range(no_samples):
            # sample gaussian noise for each weight and each bias
            weight_epsilons = Variable(self.weight_mus.data.new(self.weight_mus.size()).normal_())
            
            # calculate the weight and bias stds from the rho parameters
            weight_stds = torch.log(1 + torch.exp(self.weight_rhos))
            
            # calculate samples from the posterior from the sampled noise and mus/stds
            weight_sample = self.weight_mus + weight_epsilons*weight_stds
            
            all_samples += weight_sample.view(-1).cpu().data.numpy().tolist()
            
        return all_samples
    


###########################################################################################
# SGLD

class LangevinLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LangevinLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weights = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        self.biases = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))
        
    def forward(self, x):
        
        return torch.mm(x, self.weights) + self.biases
    

def initialize_weights_uniform(model, lower, upper):
    """
    Initializes the weights of the given model with values drawn from a uniform distribution
    between the provided lower and upper bounds.

    :param model: The PyTorch model to initialize
    :param lower: The lower bound of the uniform distribution
    :param upper: The upper bound of the uniform distribution
    """
    for module in model.modules():
        # if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        module.weight.data.uniform_(lower, upper)
        if module.bias is not None:
            module.bias.data.uniform_(lower, upper)


###########################################################################################
# AutoEncoder
class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride=1,
            padding=1,
            bias=True,
            dilation=1,
            with_bn=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))
    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs



###########################################################################################
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


###########################################################################################
###########################################################################################
# TRACKING

###########################################################################################
# METRICS POSITIONING
def compute_metrics_positioning(t, t_hat, params):

    num_agents = params.num_agents
    number_state_variables = params.number_state_variables
    dimensions = int(number_state_variables/2)

    # Group t and t_hat per single agent (discard first timestep with fictitious values) -> num_agents x num_instants x states x 1
    t = np.array([np.array([np.array(t[n][agent]).reshape(-1, 1) for n in range(1, len(t))]) for agent in range(num_agents)])
    t = t[:,:,:number_state_variables,:]
    t_hat = np.array([np.array([np.array(t_hat[n][agent]).reshape(-1, 1) for n in range(1, len(t_hat))]) for agent in range(num_agents)])

    RMSE_pos_per_agent =  {agent: np.sqrt(np.mean(np.sum((t[agent][:,:dimensions]-t_hat[agent][:,:dimensions])**2, 1))) for agent in range(num_agents)}
    RMSE_vel_per_agent = {agent: np.sqrt(np.mean(np.sum((t[agent][:,dimensions:]-t_hat[agent][:,dimensions:])**2, 1))) for agent in range(num_agents)}
    
    MAE_pos_per_agent = {agent: np.mean(np.sqrt(np.sum((t[agent][:,:dimensions]-t_hat[agent][:,:dimensions])**2, 1))) for agent in range(num_agents)}
    MAE_vel_per_agent = {agent: np.mean(np.sqrt(np.sum((t[agent][:,dimensions:]-t_hat[agent][:,dimensions:])**2, 1))) for agent in range(num_agents)}

    AE_pos_per_agent_per_timestep = {agent: np.sqrt(np.sum((t[agent][:,:dimensions]-t_hat[agent][:,:dimensions])**2, 1)) for agent in range(num_agents)}
    AE_vel_per_agent_per_timestep = {agent: np.sqrt(np.sum((t[agent][:,dimensions:]-t_hat[agent][:,dimensions:])**2, 1)) for agent in range(num_agents)}

    metrics = {'RMSE_pos_per_agent':RMSE_pos_per_agent, 
               'RMSE_vel_per_agent':RMSE_vel_per_agent,
               'MAE_pos_per_agent':MAE_pos_per_agent,
               'MAE_vel_per_agent':MAE_vel_per_agent,
               'AE_pos_per_agent_per_timestep':AE_pos_per_agent_per_timestep,
               'AE_vel_per_agent_per_timestep':AE_vel_per_agent_per_timestep,}
    
    return metrics

###########################################################################################
# EKF

def resampleSystematic(w,N):
    indx = np.zeros((N,1))
    Q = np.cumsum(w)
    T = np.linspace(0,1-1/N,N) + np.random.rand(1)/N
    T = np.append(T, 1)
    i=0
    j=0
    while i<N :
        if T[i]<Q[j]:
            indx[i]=j
            i=i+1
        else:
            j=j+1
    return indx

def regularizeAgentParticles(samples):
    # regularizationVAs = parameters.regularizationVAs
    numParticles = samples.shape[1]
    uniqueParticles = len(np.unique(samples))
    covarianceMatrix = np.cov(samples) / uniqueParticles ** (1 / 3)
    samples = samples + np.random.multivariate_normal(np.zeros((samples.shape[0],)),covarianceMatrix+1e-8*np.eye(covarianceMatrix.shape[0]),numParticles).transpose()
    return samples
   
def compute_product_Gaussian_scalar(means_vars):
    num_Gaussian = len(means_vars)
    new_var = 0
    for g in range(num_Gaussian):
        new_var += 1./np.array(means_vars[g][1])
    new_var = np.reciprocal(new_var)

    new_mean = 0
    for g in range(num_Gaussian):
        new_mean += np.array(means_vars[g][0])/np.array(means_vars[g][1])
    new_mean = new_var*new_mean
    # new_mean = new_var*(mean_1/var_1 + mean_2/var_2)
    return [new_mean, new_var]

def compute_product_Gaussian_vector(means_vars):
    num_Gaussian = len(means_vars)
    inv_covariances = []
    for g in range(num_Gaussian):
        inv_covariances.append(np.linalg.inv(np.array(means_vars[g][1])))
    inv_covariances = np.array(inv_covariances)
    new_var = np.linalg.inv(np.sum(inv_covariances, 0))

    new_mean = []
    for g in range(num_Gaussian):
        new_mean.append(inv_covariances[g]@np.array(means_vars[g][0]))
    new_mean = np.array(new_mean)
    new_mean = new_var@np.sum(new_mean, 0)
    return [new_mean, new_var]


def compute_H(u, s, dim, method):
    u = u.reshape(dim, 1)[:dim]
    s = s.reshape(-1, dim)[:,:dim]
    N = s.shape[0]
    
    def compute_a(s, u):
        s = s.reshape(dim, 1)
        u = u.reshape(dim, 1)
        d_i = np.linalg.norm(s - u)
        return (s - u) / d_i

    if method.lower() == 'rtt' or method.lower() == 'toa':
        H = np.zeros((N, dim))
        for i in range(N):
            H[i, :] = -compute_a(s[i, :], u).squeeze()
            
    elif method.lower() == 'tdoa':
        a = np.zeros((N, dim))
        H = np.zeros((N - 1, dim))
        for i in range(N):
            a[i, :] = compute_a(s[i, :], u).squeeze()
        for i in range(1, N):
            H[i - 1, :] = a[0, :] - a[i, :]

    elif method.lower() == 'aod':
        d_h = np.zeros(N)
        d = np.zeros((N, dim))
        H = np.zeros((N * (dim - 1), dim))
        for i in range(N):
            d[i, :] = (s[i, :dim].reshape(-1, 1) - u[:dim].reshape(-1, 1)).squeeze()
            d_h[i] = np.linalg.norm(d[i, :2])
        for i in range(N):
            H[i, 0] = d[i, 1] / (d_h[i] ** 2)
            H[i, 1] = -d[i, 0] / (d_h[i] ** 2)
        if dim == 3:
            for i in range(N):
                j = i + N
                d_square = (np.linalg.norm(d[i, :])) ** 2
                H[j, 0] = (d[i, 2] * d[i, 0]) / (d_h[i] * d_square)
                H[j, 1] = (d[i, 2] * d[i, 1]) / (d_h[i] * d_square)
                H[j, 2] = -d_h[i] / d_square
    
    # Implement other methods like 'RTT+AoD', 'TDoA+AoD' similarly
    
    return H


def compute_h(u, s, dim, method):
    u = u.reshape(dim, 1)[:dim]
    s = s.reshape(-1, dim)[:,:dim]
    N = s.shape[0]

    if method.lower() == 'rtt' or method.lower() == 'toa':
        h = np.zeros((N, 1))
        for i in range(N):
            h[i] = np.linalg.norm(s[i, :].reshape(-1, 1) - u)
    elif method.lower() == 'aod':
        h = np.zeros((N, 2))  # N x 2 matrix for azimuth and elevation
        for i in range(N):
            dx = s[i, 0] - u[0]
            dy = s[i, 1] - u[1]
            dz = s[i, 2] - u[2] if dim == 3 else 0  # Assuming z-coordinate exists if dim==3
            r = np.linalg.norm([dx, dy, dz])

            azimuth = np.rad2deg(np.arctan2(dy, dx))
            elevation = np.rad2deg(np.arctan2(dz, np.sqrt(dx**2 + dy**2)))

            h[i, 0] = azimuth
            h[i, 1] = elevation

    return h




def Non_linear_LS_TOA(rho, u_0, s, K, dim):

    rho = rho.reshape(-1, 1)
    s = s.reshape(-1, dim)

    u_k = u_0[:dim].reshape(-1, 1)
    mu_0 = 0.01
    mu_k = mu_0
    
    num_meas = len(rho)

    diag_R = np.ones(num_meas)

    R = np.diag(diag_R)

    def compute_H(s, u, dim):
        N = s.shape[0]
        a = np.zeros((N, dim))
        H = np.zeros((N, dim))
        for i in range(N):
            a[i, :] = compute_a(s[i, :].reshape(-1, 1), u).squeeze()
        for i in range(N):
            H[i, :] = -a[i, :]
        return H
    
    def compute_a(s, u):
        d = np.linalg.norm(s - u)
        return (s - u) / d
    
    def compute_delta_rho(s, rho, u):
        N = s.shape[0]
        d = np.zeros((N, 1))
        for i in range(N):
            d[i] = np.linalg.norm(s[i, :].reshape(-1, 1) - u)
        delta_rho = rho - d
        return delta_rho

    for k in range(K):
        delta_rho = compute_delta_rho(s, rho, u_k)
        H = compute_H(s, u_k, dim)
        delta_u = np.linalg.inv(H.T @ np.linalg.inv(R) @ H) @ H.T @ np.linalg.inv(R) @ delta_rho
        u_k = u_k + mu_k * delta_u

    return u_k



def LS_TOA(rho, s, dim):
    
    def compute_a(s, u):
        s = s.reshape(dim, 1)
        u = u.reshape(dim, 1)
        d_i = np.linalg.norm(s - u)
        return (s - u) / d_i
    
    def delta_rho(u, rho, s):
        computed_rho = np.zeros(N)
        for i in range(N):
            computed_rho[i] = np.linalg.norm(s[i, :] - u)
        return computed_rho - rho
    
    rho = rho.reshape(-1, 1)
    s = s.reshape(-1, dim)
    N = s.shape[0]

    H = np.zeros((N, dim))
    for i in range(N):
        H[i, :] = compute_a(s[i, :], u).squeeze()
    rho = delta_rho(rho, s)
    x = np.linalg.inv(H.T @ H) @ H.T @ rho
    u = x + s[0, :]
    
    return u