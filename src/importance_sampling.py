# MIT License

# Copyright (c) 2021 Louis Popi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Libraries
from functools import partial
import tensorflow_probability.substrates.jax as tfp
import jax
import jax.numpy as jnp
import jax.scipy as jsc
tfd = tfp.distributions

class ClassicIS:

  def __init__(self, n_samples, prior, importance_distr):
    self.n_samples = n_samples
    self.prior = prior
    self.importance_distr = importance_distr
    self.use_extended_prior = False
  
  @partial(jax.jit, static_argnums=(0,))
  def weights(self, samples):
     return self.prior.log_prob(samples) - self.importance_distr.log_prob(samples)

  def log_estimate(self, loglikelihood, seed):
    # Sample from the importance distribution
    samples = self.importance_distr.sample((self.n_samples, ), seed=seed)
    # Update the key
    _, seed_next = jax.random.split(seed)
    # Compute the weights
    weights = self.weights(samples)
    # Compute the MC estimator
    log_Z = jsc.special.logsumexp(weights + loglikelihood(samples)) - jnp.log(self.n_samples)
    return log_Z, seed_next

  def estimate(self, loglikelihood, seed):
    log_Z, seed_next = self.log_estimate(loglikelihood, seed)
    return jnp.exp(log_Z), seed_next

  @partial(jax.jit, static_argnums=(0,2))
  def log_estimate_gibbs_correlated(self, z0, loglikelihood, seed, step_size=pow(10,-2)):
    # Sample a trajectory from the proposal
    z = tfp.mcmc.sample_chain(
        num_results=self.n_samples-1,
        current_state=z0,
        kernel=tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=self.importance_distr.log_prob,
            new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=step_size)),
      trace_fn=None,
      seed=seed)
    # Update the key
    _, seed_next = jax.random.split(seed)
    # Compute the weights
    w = self.weights(z) + loglikelihood(z)
    return w, z, seed_next

 # NOTE: Hardcoded bar(omega) = 1_{[0,K]}

class NEOIS(ClassicIS):

  def __init__(self, n_samples, prior, importance_distr, transformation, momentum_scale, K):
    super().__init__(n_samples, prior, importance_distr)
    self.transformation = transformation
    self.momentum_dist = tfp.distributions.MultivariateNormalDiag(loc=jnp.zeros(transformation.Mdim), scale_identity_multiplier=momentum_scale)
    self.K = K
    self.use_extended_prior = True

  @partial(jax.jit, static_argnums=(0,1,4))
  def weights(self, loglikelihood, q0, p0, save_traj=False):
    # Extend the prior distribution
    extended_prior_log_prob = jax.jit(lambda q,p : self.importance_distr.log_prob(q) + self.momentum_dist.log_prob(p))
    # Get the push-forward and push-backward
    if save_traj:
      positions_f, momentums_f, w_is, Lq, push_f, push_b = self.transformation.compute_pushfwb(loglikelihood, extended_prior_log_prob, self.prior.log_prob,
                                            self.importance_distr.log_prob, q0, p0, jnp.zeros((q0.shape[0],)), self.K, extract_forward_traj=True)
    else:
      w_is, Lq, push_f, push_b = self.transformation.compute_pushfwb(loglikelihood, extended_prior_log_prob, self.prior.log_prob,
                                                                   self.importance_distr.log_prob, q0, p0, jnp.zeros((q0.shape[0],)), self.K)
    # Compute the weights
    idx_array = jnp.arange(self.K)
    denum = jnp.array([jnp.vstack([push_f[k-idx_array[:k]], push_b[idx_array[k:]-k]]) for k in range(self.K)])
    weights = push_f - jsc.special.logsumexp(denum, axis=1)
    # Reweight for the importance distribution
    weights += w_is
    # Returns
    if save_traj:
      return weights + Lq, (positions_f, momentums_f)
    else:
      # Compute the estimator
      z_hat = jsc.special.logsumexp(weights + Lq, axis=1)
      return z_hat

  # Vectorize over K for weights computation for SIR
  # NOTE: Requires some vmap tricks
  @partial(jax.vmap, in_axes=(None, None, 1, 1))
  def weights_save_traj(self, b, c, d):
    return self.weights(b,c,d,save_traj=True)
  def weights_v(self, b, c, d):
    z_hat, (positions_f, momentums_f) = self.weights_save_traj(b,c,d)
    return jnp.transpose(z_hat, axes=(1,2,0)), (jnp.transpose(positions_f, axes=(1,2,0,3)), jnp.transpose(momentums_f, axes=(1,2,0,3)))

  def log_estimate(self, loglikelihood, seed):
    # Sample from the extended prior
    seed_q, seed_p, seed_next = jax.random.split(seed, num=3) # Update the key
    q0 = self.importance_distr.sample((self.n_samples, ), seed=seed_q)
    p0 = self.momentum_dist.sample((self.n_samples, ), seed=seed_p)
    _, seed_next = jax.random.split(seed_next) # Update the key
    # Compute the final estimator and weights
    z_hat = self.weights(loglikelihood, q0, p0, save_traj=False)
    z = jsc.special.logsumexp(z_hat) - jnp.log(self.n_samples)
    return z, seed_next
  
  def estimate_E_T(self, loglikelihood, log_real_constant, seed):
    # Sample from the extended prior
    seed_q, seed_p, seed_next = jax.random.split(seed, num=3) # Update the key
    q0 = self.importance_distr.sample((self.n_samples, ), seed=seed_q)
    p0 = self.momentum_dist.sample((self.n_samples, ), seed=seed_p)
    _, seed_next = jax.random.split(seed_next) # Update the key
    # Compute the final estimator and weights
    z_hat = self.weights(loglikelihood, q0, p0, save_traj=False)
    z = jsc.special.logsumexp(z_hat) - jnp.log(self.n_samples)
    # Divide by the real constant and square
    return z, jsc.special.logsumexp(2.0 * (z_hat - log_real_constant)) - jnp.log(self.n_samples), seed_next

  @partial(jax.jit, static_argnums=(0,2))
  def log_estimate_gibbs_correlated(self, z0, loglikelihood, seed, step_size=pow(10,-1)):
    # Update the key
    seed_q, seed_p, seed_next = jax.random.split(seed, num=3)
    # Sample a trajectory of positions from the proposal
    # Unpack the extended dist
    q0, p0 = z0
    q0, p0 = q0[0], p0[0] # Take the first K
    # Sample
    q = tfp.mcmc.sample_chain(
        num_results=self.n_samples-1,
        current_state=q0,
        kernel=tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=self.importance_distr.log_prob,
            new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=step_size)),
      trace_fn=None,
      seed=seed_q)
    # Sample a momentum
    p = jax.random.normal(key=seed_p, shape=q.shape)
    # Compute the weights
    z_hat, trajectory = self.weights_v(loglikelihood, q, p)
    return z_hat, trajectory, seed_next