import numpy as np

class AdEMAMix:
    def __init__(self, theta_0, T, eta, epsilon, beta1, beta2, lamb, beta3, alpha, T_alpha_beta3):
        self.theta = theta_0  # Initial model parameters (weights)
        self.T = T  # Number of iterations
        self.eta = eta  # Learning rate
        self.epsilon = epsilon  # Small constant to avoid division by zero
        self.beta1 = beta1  # AdamW parameter (fast EMA decay rate)
        self.beta2 = beta2  # AdamW parameter (second moment decay rate)
        self.lamb = lamb  # Weight decay term (similar to L2 regularization)
        self.beta3 = beta3  # AdEMAMix parameter (slow EMA decay rate)
        self.alpha = alpha  # AdEMAMix parameter (mixing coefficient between m1 and m2)
        self.T_alpha_beta3 = T_alpha_beta3  # Warmup period for beta3 and alpha
        self.beta_start = beta1  # Initialize beta3 warmup with beta1 value
        self.t = 0
        self.m1 = np.zeros_like(self.theta)  # First moment estimate
        self.m2 = np.zeros_like(self.theta)  # Slow EMA
        self.v = np.zeros_like(self.theta)   # Second moment estimate (variance)

    def f_beta3(self, t):
        """Linear warmup scheduler for beta3."""
        if t >= self.T_alpha_beta3:
            return self.beta3
        else:
            return self.beta_start + (self.beta3 - self.beta_start) * (t / self.T_alpha_beta3)

    def f_alpha(self, t):
        """Linear warmup scheduler for alpha."""
        if t >= self.T_alpha_beta3:
            return self.alpha
        else:
            return self.alpha * (t / self.T_alpha_beta3)

    def optimize(self, grad):
        """Perform one optimization step."""
        self.t += 1
        
        beta3_t = self.f_beta3(self.t)
        alpha_t = self.f_alpha(self.t)
        
        # Update fast EMA (m1)
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * grad
        
        # Update slow EMA (m2)
        self.m2 = beta3_t * self.m2 + (1 - beta3_t) * grad
        
        # Update second moment estimate (v)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # Apply bias corrections to m1 and v
        m1_hat = self.m1 / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update model parameters (theta)
        self.theta -= self.eta * ((m1_hat + alpha_t * self.m2) / (np.sqrt(v_hat) + self.epsilon) + self.lamb * self.theta)

        return self.theta