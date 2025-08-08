import numpy as np
from scipy.stats import invgamma, lognorm
# from .cython_code.ABC_CAPF import ABC_CAPF
from .ABC_CAPF import ABC_CAPF
class ABC_pg_cAPF:
    
    def __init__(self, observations, burn_in, sample, particle_count, epsilon, Z_params):
        self.observations = observations
        self.burn_in = burn_in
        self.sample = sample
        self.particle_count = particle_count
        self.epsilon = epsilon
        self.Z_params = Z_params
        
        
    def draw_initial_params(self):
        a_0, b_0 = 2, 0.5 
        sigma_h2 = invgamma.rvs(a_0, scale=b_0)
        
        mu_0 = np.array([[0], [0.9]])
        mu_0_flat = mu_0.flatten()
        
        lambda_0 = np.array([[1, 0], [0, 1]])  # Precision matrix (inverse of covariance matrix)

        tau, phi = np.random.multivariate_normal(mu_0_flat, sigma_h2 * np.linalg.inv(lambda_0))

        # Ensure phi is within the required bounds
        while abs(phi) >= 0.995:
            tau, phi = np.random.multivariate_normal(mu_0_flat, sigma_h2 * np.linalg.inv(lambda_0))

        return sigma_h2, tau, phi, a_0, b_0, mu_0, lambda_0

    def calculate_input_trajectory(self, observations, sigma_h2, tau, phi):

        mean_log = tau / (1 - phi)  # Mean of the logarithm of the variable
        sigma_log = np.sqrt(sigma_h2 / (1 - phi**2))  # Standard deviation of the logarithm of the variable
        # Sample from the lognormal distribution
        input_trajectory = np.random.lognormal(mean=mean_log, sigma=sigma_log, size=len(observations)+1)
        return input_trajectory
    
    def compute_lambda_t(self, X, lambda_0):
        # Calculate the transpose of X
        XT = X.T
        # Perform matrix multiplication
        XTX = XT @ X
        lambda_t = XTX + lambda_0
        return lambda_t
    
    def compute_mu_t(self, mu_0, lambda_0, X, y, lambda_t):

        # Perform matrix inversion using float64
        term1 = np.linalg.inv(lambda_t)
        term2 = (lambda_0 @ mu_0) 
        term2 = term2 + (X.T @ y)
        mu_t = term1 @ term2
        return mu_t


    def compute_b_t(self, b_0, y, lambda_0, mu_0, mu_t, lambda_t):
        term1 = y.T @ y
        
        term2 = mu_0.T @ lambda_0 @ mu_0
        
        term3 = mu_t.T @ lambda_t @ mu_t

        b_t = (b_0 + 0.5 * (term1 + term2 - term3)).item()

        return b_t
    
    def sample_parameters(self, a, b, mu, Lambda):
        # Sample sigma_h^2 from the inverse gamma distribution
        sigma_h2 = invgamma.rvs(a, scale=b)

        mu_flat = mu.flatten()

        # Perform the operation using float64 precision

        inv_Lambda = np.linalg.inv(Lambda)
        
        tau_phi_covariance = sigma_h2 * inv_Lambda

        # Sample from the multivariate normal distribution
        tau, phi = np.random.multivariate_normal(mu_flat, tau_phi_covariance)

        # Ensure phi is within the required bounds, adjusting if necessary
        while abs(phi) >= 0.995:
            tau, phi = np.random.multivariate_normal(mu_flat, tau_phi_covariance)

        return sigma_h2, tau, phi

    
    def run(self):
        sigma_h2, tau, phi, a_0, b_0, mu_0, lambda_0  = self.draw_initial_params()
        input_trajectory = self.calculate_input_trajectory(self.observations, sigma_h2, tau, phi )
        epsilon = self.epsilon
        parameter_samples = []
        conditional_probabilities = []

        total_iterations = self.burn_in + self.sample
        

        for t in range(total_iterations + 1):
            abc_capf = ABC_CAPF(self.observations, tau, phi, sigma_h2, input_trajectory, epsilon, self.Z_params, self.particle_count)
#             cProfile.runctx('abc_capf.run()', globals(), locals(), filename=None)
    
            input_trajectory = abc_capf.run()
            input_trajectory = np.array(input_trajectory, dtype=np.float64)
            X = np.column_stack((np.ones(len(input_trajectory)-1), np.log(input_trajectory[:-1])))
            y = np.log(input_trajectory[1:]).reshape(-1, 1)
            lambda_t = self.compute_lambda_t(X, lambda_0)
            mu_t = self.compute_mu_t(mu_0, lambda_0, X, y, lambda_t)
            a_t = a_0 + len(self.observations)/2

            b_t = self.compute_b_t(b_0, y, lambda_0, mu_0, mu_t, lambda_t)

            sigma_h2, tau, phi = self.sample_parameters(a_t, b_t, mu_t, lambda_t)

            # print(t, tau, phi, sigma_h2)

            # if t > self.burn_in:
            #         # parameter_samples.append({
            #         #     'iteration': t,
            #         #     'sigma_h2': sigma_h2,
            #         #     'tau': tau,
            #         #     'phi': phi
            #         # })
            #     mu = tau + phi * np.log(input_trajectory[-2])

            #     std_dev = np.sqrt(sigma_h2)
                
            #     scale = np.exp(mu)

            #     if scale > 0:
            #         conditional_probability = lognorm.cdf(input_trajectory[-1], s=std_dev, scale=scale)

            #         conditional_probabilities.append(conditional_probability)


        h_prev = input_trajectory
        sigma = np.sqrt(sigma_h2)
        
        return h_prev, tau, phi, sigma, #conditional_probabilities