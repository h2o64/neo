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

class Transformation:

  # Compute the pushforward and pushbackward at a given step
  # NOTE: Only computes the K first values
  def compute_pushfwb(self, likelihood, extended_prior_log_prob, prior_log_prob, importance_log_prob, q0, p0, log_jac_0, K, extract_forward_traj=False):
    # Step of the push process
    @jax.jit
    def compute_pushfwb_step(carry, x):
      # Unpack the carry
      q_f, p_f, log_jac_f, q_b, p_b, log_jac_b = carry
      # Compute the classic IS weights to go from the importance density to proposal density
      w_is = prior_log_prob(q_f) - importance_log_prob(q_f)
      # Propagate positions through the likelihood
      Lq = likelihood(q_f)
      # Compute the push push forward / backward
      push_f = extended_prior_log_prob(q_f, p_f) + log_jac_f
      push_b = extended_prior_log_prob(q_b, p_b) + log_jac_b
      # Forward pass
      q_f_next, p_f_next, log_jac_f_next = self.forward(q_f, p_f)
      log_jac_f_cur = log_jac_f_next + log_jac_f
      # Backward pass
      q_b_next, p_b_next, log_jac_b_next = self.backward(q_b, p_b)
      log_jac_b_cur = log_jac_b_next + log_jac_b
      # Return the result and next carry
      if extract_forward_traj:
        return (q_f_next, p_f_next, log_jac_f_cur, q_b_next, p_b_next, log_jac_b_cur), (q_f, p_f, w_is, Lq, push_f, push_b)
      else:
        return (q_f_next, p_f_next, log_jac_f_cur, q_b_next, p_b_next, log_jac_b_cur), (w_is, Lq, push_f, push_b)
    # Compute it all with jax.scan
    _, results = jax.lax.scan(compute_pushfwb_step, (q0, p0, log_jac_0, q0, p0, -log_jac_0), jnp.arange(K))
    return results

class DampedHamiltonianEuler(Transformation):

  def __init__(self, gamma, dt, grad_U, Mdim, Mscale):
    self.gamma = gamma
    self.dt = dt
    self.grad_U = grad_U
    self.Mdim = Mdim
    self.Mscale = Mscale
    self.Minv= (1.0/Mscale) * jnp.eye(Mdim)
  
  # Compute the jacobian
  def get_log_jacobian(self, q, p, forward):
    if forward:
      return -1.0 * p.shape[1] * self.gamma * self.dt * jnp.ones((p.shape[0],))
    else:
      return p.shape[1] * self.gamma * self.dt * jnp.ones((p.shape[0],))

  # Forward order one mapping
  @partial(jax.jit, static_argnums=(0,))
  def forward(self, q, p):
    p_next = jnp.exp(- self.gamma * self.dt) * p
    p_next = p_next - self.dt * self.grad_U(q)
    q_next = q + self.dt * jnp.matmul(p_next, self.Minv)
    return q_next, p_next, self.get_log_jacobian(q, p, True)

  # Backward order one mapping
  @partial(jax.jit, static_argnums=(0,))
  def backward(self, q, p):
    q_prev = q - self.dt * jnp.matmul(p, self.Minv)
    p_prev = jnp.exp(self.gamma * self.dt) * p
    p_prev = p_prev + self.dt * jnp.exp(self.gamma * self.dt) * self.grad_U(q_prev)
    return q_prev, p_prev, self.get_log_jacobian(q, p, False)

class DampedHamiltonianLeapFrog(Transformation):

  def __init__(self, gamma, dt, grad_U, Mdim, Mscale):
    self.gamma = gamma
    self.dt = dt
    self.grad_U = grad_U
    self.Mdim = Mdim
    self.Mscale = Mscale
    self.Minv= (1.0/Mscale) * jnp.eye(Mdim)

  # Compute the jacobian
  def get_log_jacobian(self, q, p, forward):
    if forward:
      return p.shape[1] * self.gamma * self.dt * jnp.ones((p.shape[0],))
    else:
      return -1.0 * p.shape[1] * self.gamma * self.dt * jnp.ones((p.shape[0],))

  # Forward order two mapping
  @partial(jax.jit, static_argnums=(0,))
  def forward(self, q, p):
    q_next = q + self.dt/2.0 * jnp.exp(-self.gamma * self.dt / 2.0) * jnp.matmul(p, self.Minv)
    p_next = jnp.exp(-self.gamma * self.dt / 2.0) * p
    p_next = p_next - self.dt * self.grad_U(q_next)
    q_next = q + self.dt/2.0 * jnp.matmul(p, self.Minv)
    p_next *= jnp.exp(-self.gamma * self.dt/2.0)
    return q_next, p_next, self.get_log_jacobian(q, p, True)

  # Backward order two mapping
  @partial(jax.jit, static_argnums=(0,))
  def backward(self, q, p):
    p_prev = jnp.exp(self.gamma * self.dt/2.0) * p
    q_prev = q - self.dt/2.0 * jnp.matmul(p_prev, self.Minv)
    p_prev = (p_prev + self.dt * self.grad_U(q_prev)) * jnp.exp(self.gamma * self.dt/2.0)
    q_prev = q_prev - self.dt/2.0 * jnp.exp(self.gamma * self.dt/2.0) * jnp.matmul(p_prev, self.Minv)
    return q_prev, p_prev, self.get_log_jacobian(q, p, False)