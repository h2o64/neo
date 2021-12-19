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
import jax
import jax.numpy as jnp
import jax.scipy as jsc

class iSIR:

  def __init__(self, is_estimator):
    self.estimator = is_estimator
    self.use_extended_prior = is_estimator.use_extended_prior
  
  @partial(jax.jit, static_argnums=(0,4))
  def sample_step(self, traj, weights, idx, loglikelihood, seed):
    # Unpack idx
    if self.use_extended_prior:
      i, k = idx
    else:
      i = idx
    # Extract the i-th trajectory and select the first chain
    if self.use_extended_prior:
      traj_cur = (traj[0][:,i,0], traj[1][:,i,0])
      weights_cur = weights[:,i,0]
    else:
      traj_cur = traj[i,0]
      weights_cur = weights[i,0]
    # Compute the weights
    weights_new, traj_new, seed_next = self.estimator.log_estimate_gibbs_correlated(traj_cur, loglikelihood, seed)
    # Concatenate everything
    if self.use_extended_prior:
      weights_tot = jnp.hstack([jnp.expand_dims(weights_cur, axis=1), weights_new])
      traj_tot_q = jnp.hstack([jnp.expand_dims(traj_cur[0], axis=1), traj_new[0]])
      traj_tot_p = jnp.hstack([jnp.expand_dims(traj_cur[1], axis=1), traj_new[1]])
      traj_tot = (traj_tot_q, traj_tot_p)
    else:
      weights_tot = jnp.vstack([jnp.expand_dims(weights_cur, axis=0), weights_new])
      traj_tot = jnp.vstack([jnp.expand_dims(traj_cur, axis=0), traj_new])
    # Compute the indexes
    if self.use_extended_prior:
      est_traj = jsc.special.logsumexp(weights_tot, axis=0)
    else:
      est_traj = weights_tot
    i_new = jax.random.categorical(key=seed_next, logits=est_traj.T)
    # Update the key
    _, seed_next = jax.random.split(seed_next)
    # Select the trajectory
    if self.use_extended_prior:
      traj_cur_new = (traj_tot[0][:,i_new,0], traj_tot[1][:,i_new,0])
      weights_cur_new = weights_tot[:,i_new,0]
    else:
      traj_cur_new = traj_tot[i_new,0]
      weights_cur_new = weights_tot[i_new,0]
    # Return the trajectory
    if self.use_extended_prior:
      # Sample from K
      k_new = jax.random.categorical(key=seed_next, logits=weights_cur_new.T)
      # Update the key
      _, seed_next = jax.random.split(seed_next)
      # Return
      return traj_tot, weights_tot, (i_new, k_new), (traj_cur_new[0][k_new][:,0], traj_cur_new[1][k_new][:,0]), seed_next
    else:
      return traj_tot, weights_tot, i_new, traj_cur_new, seed_next
    
  def chain_sample(self, n, n_chain, loglikelihood, seed):
    # Initial trajectory
    if self.use_extended_prior:
      seed_q, seed_p, seed_next = jax.random.split(seed, num=3)
      q = self.estimator.importance_distr.sample((self.estimator.K, self.estimator.n_samples, n_chain, ), seed=seed_q)
      p = self.estimator.momentum_dist.sample(q.shape[:-1], seed=seed_p)
      init_traj = (q,p)
    else:
      seed_q, seed_next = jax.random.split(seed)
      init_traj = self.estimator.importance_distr.sample((self.estimator.n_samples, n_chain, ), seed=seed_q)
    # Initial indexes set at 0
    i = jnp.zeros(n_chain, dtype=int)
    if self.use_extended_prior:
      k = jnp.zeros(n_chain, dtype=int)
      init_idx = (i,k)
    else:
      init_idx = i
    # Initialize weights
    if self.use_extended_prior:
      init_weights = jnp.log(jnp.zeros((self.estimator.K, self.estimator.n_samples, n_chain)))
    else:
      init_weights = jnp.log(jnp.zeros((self.estimator.n_samples, n_chain)))
    # Recurrent call
    def sampling_step_rec(carry, x):
      # Unpack the carry
      traj, weights, idx, cur_seed = carry
      # Sample
      traj_new, weights_new, idx_new, y, seed_next = self.sample_step(traj, weights, idx, loglikelihood, cur_seed)
      return (traj_new, weights_new, idx_new, seed_next), y
    _, samples = jax.lax.scan(sampling_step_rec, (init_traj, init_weights, init_idx, seed_next), jnp.arange(n))
    # Return proudly made samples
    return samples