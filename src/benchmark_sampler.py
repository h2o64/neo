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
import tensorflow_probability.substrates.jax as tfp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from importance_sampling import ClassicIS, NEOIS
from isir import iSIR

# Perform a benchmark
def benchmark_samplers(is_params, neo_params, transformation_params, distributions, sampling_params, loglikelihood, seed):

  # Modify IS params
  is_params['n_samples'] = sampling_parameters['n_samples']

  # Classic i-SIR
  classic_is = ClassicIS(**is_params)
  classic_sampler = iSIR(classic_is)
  q_classic_is = classic_sampler.chain_sample(sampling_parameters['n'], sampling_parameters['n_chain'], loglikelihood=loglikelihood, seed=seed)
  q_classic_is = q_classic_is.reshape((q_classic_is.shape[0] * q_classic_is.shape[1], q_classic_is.shape[2]))

  # NEO-MCMC
  transformation = DampedHamiltonianEuler(**transformation_params)
  neo_is = NEOIS(**is_params, **neo_params, transformation=transformation)
  neo_sampler = iSIR(neo_is)
  q_neo_is, _ = neo_sampler.chain_sample(sampling_parameters['n'], sampling_parameters['n_chain'], loglikelihood=loglikelihood, seed=seed)
  q_neo_is = q_neo_is.reshape((q_neo_is.shape[0] * q_neo_is.shape[1], q_neo_is.shape[2]))

  # RW-MH MCMC
  q_mh = tfp.mcmc.sample_chain(
      num_results=sampling_params['n'],
      current_state=jnp.zeros((sampling_params['n_chain'], 2)),
      kernel=tfp.mcmc.RandomWalkMetropolis(
          target_log_prob_fn=distributions['target'].log_prob,
          new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=0.1)),
      seed=seed,
      trace_fn=None)
  q_mh = q_mh.reshape((q_mh.shape[0] * q_mh.shape[1], q_mh.shape[2]))

  # MALA
  q_mala = tfp.mcmc.sample_chain(
      num_results=sampling_params['n'],
      current_state=jnp.zeros((sampling_params['n_chain'], 2)),
      kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
          target_log_prob_fn=distributions['target'].log_prob,
          step_size=0.1),
      seed=seed,
      trace_fn=None)
  q_mala = q_mala.reshape((q_mala.shape[0] * q_mala.shape[1], q_mala.shape[2]))

  # HMC
  q_hmc = tfp.mcmc.sample_chain(
      num_results=sampling_params['n'],
      current_state=jnp.zeros((sampling_params['n_chain'], 2)),
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=distributions['target'].log_prob,
          step_size=0.1,
          num_leapfrog_steps=2),
      seed=seed,
      trace_fn=None)
  q_hmc = q_hmc.reshape((q_hmc.shape[0] * q_hmc.shape[1], q_hmc.shape[2]))

  # Return everything
  return q_classic_is, q_neo_is, q_mh, q_mala, q_hmc

# Plot everything
def plot_samples(target, q_classic_is, q_neo_is, q_mh, q_mala, q_hmc, grid_lim, nb_points=1000, alpha_plots=0.7, alpha_td=1.0, scenario_num=1):
  # Colors
  colors = ['pink','red','green','orange','purple']
  # Evaluate the target density
  grid_plot = (-grid_lim, grid_lim, -grid_lim, grid_lim)
  xplot = jnp.linspace(-grid_lim, grid_lim, nb_points)
  yplot = jnp.linspace(-grid_lim, grid_lim, nb_points)
  Xplot, Yplot = jnp.meshgrid(xplot, yplot)
  td = target.prob(jnp.dstack([Xplot, Yplot]))
  # Figure size
  plt.figure(figsize=(30,20), facecolor='white')
  # Plot the target density
  plt.subplot(2, 3, 1)
  plt.imshow(td, alpha=alpha_td, extent=grid_plot, cmap='Blues', origin='top')
  plt.title("Target density (Scenario {})".format(scenario_num))
  # Classic i-SIR
  plt.subplot(2, 3, 2)
  plt.imshow(td, alpha=alpha_td, extent=grid_plot, cmap='Blues', origin='top')
  plt.scatter(q_classic_is[:,0], q_classic_is[:,1], label="Classic i-SIR", color=colors[0], alpha=alpha_plots)
  plt.title("Classic i-SIR")
  # NEO-MCMC
  plt.subplot(2, 3, 3)
  plt.imshow(td, alpha=alpha_td, extent=grid_plot, cmap='Blues', origin='top')
  plt.scatter(q_neo_is[:,0], q_neo_is[:,1], label="NEO-MCMC", color=colors[1], alpha=alpha_plots)
  plt.title("NEO-MCMC")
  # Random Walk Metropolis Hastings
  plt.subplot(2, 3, 4)
  plt.imshow(td, alpha=alpha_td, extent=grid_plot, cmap='Blues', origin='top')
  plt.scatter(q_mh[:,0], q_mh[:,1], label="RW-MH", color=colors[2], alpha=alpha_plots)
  plt.title("Random Walk Metropolis Hastings")
  # Metropolis-Adjusted Langevin Algorithm
  plt.subplot(2, 3, 5)
  plt.imshow(td, alpha=alpha_td, extent=grid_plot, cmap='Blues', origin='top')
  plt.scatter(q_mala[:,0], q_mala[:,1], label="MALA", color=colors[3], alpha=alpha_plots)
  plt.title("Metropolis-Adjusted Langevin Algorithm")
  # Hamiltonian Monte Carlo
  plt.subplot(2, 3, 6)
  plt.imshow(td, alpha=alpha_td, extent=grid_plot, cmap='Blues', origin='top')
  plt.scatter(q_hmc[:,0], q_hmc[:,1], label="HMC", color=colors[4], alpha=alpha_plots)
  plt.title("Hamiltonian Monte Carlo")
  # Show
  plt.show()