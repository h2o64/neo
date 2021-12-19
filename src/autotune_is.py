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
from itertools import groupby
from sklearn.model_selection import ParameterGrid, ParameterSampler
from tqdm import tqdm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from importance_sampling import NEOIS

def list_of_dicts_to_dict_of_lists(l, batch_size):
  return {k: jnp.array_split(jnp.array([dic[k] for dic in l]), batch_size) for k in l[0]}

def dict_of_lists_to_list_of_dicts(d):
  return [dict(zip(d,t)) for t in zip(*d.values())]

# Plot the results of the autotuning
def plot_autotune_results(r, log_target_Z=jnp.log(15), plot_E_T=True):
  # Remove nan values
  r_filtered = list(filter(lambda x : not jnp.isnan(x['Z']) or not jnp.isnan(x['E_T']), r))
  # Browse each category of parameter
  for param_cat, cat_name in [('neo_params', 'NEO-IS'),('transformation_params', 'transformation')]:
    # Browse each parameter in this category
    for param in r[0][param_cat].keys():
      # Extract only this param and Z
      extracted = list(map(lambda x : (x['Z'], x['E_T'], x[param_cat][param]), r_filtered))
      # Sort them by parameters
      param_key = lambda x : x[-1]
      extracted.sort(key=param_key)
      # Setup the plot
      if plot_E_T:
        values_to_plot = [('Z', 'blue', '"Estimated Z" / "Real Z"', 0), ('E_T','green',"E_T",1)]
      else:
        values_to_plot = [('Z','blue',0)]
      # Get axis
      fig, ax1 = plt.subplots(figsize=(20,10))
      ax2 = ax1.twinx()
      axes = [ax1,ax2]
      for value_to_plot, value_color, value_label, value_idx in values_to_plot:
        # Extract means and standard deviation
        values = []
        means = []
        stds = []
        lengths = []
        for k, group in groupby(extracted, param_key):
          if values_to_plot == 'Z':
            v = jnp.log(jnp.array(list(map(lambda x : x[0], group)))) - log_target_Z
          else:
            v = jnp.log(jnp.array(list(map(lambda x : x[1], group))))
          values.append(k)
          means.append(jnp.mean(v))
          stds.append(jnp.std(v))
          lengths.append(len(v))
        # Convert to arrays
        values = jnp.array(values)
        means = jnp.array(means)
        stds = jnp.array(stds)
        lengths = jnp.array(lengths)
        # Plot everything
        axes[value_idx].plot(values, means, color=value_color, label=value_label)
        if value_to_plot == 'Z':
          axes[value_idx].axhline(y=0, linestyle='--', label="Real normalizing constant")
        axes[value_idx].fill_between(values, means - stds / jnp.sqrt(lengths), means + stds / jnp.sqrt(lengths), alpha=0.2, color=value_color)
        if value_to_plot == 'Z':
          axes[value_idx].set_xlabel("Value of {}".format(param))
        axes[value_idx].set_ylabel("Log scale")
      # Hangle legend
      h1, l1 = ax1.get_legend_handles_labels()
      h2, l2 = ax2.get_legend_handles_labels()
      ax1.legend(h1+h2, l1+l2, loc=2)
      plt.title("Error of estimation of the normalizing constant for the parameter {} ({} parameters)".format(param, cat_name))
      plt.show()

# Autotune NEO-IS
def autotune_neo(transformation_class, transformation_ranges, neo_ranges, n_samples, distributions, K, dt, dim, key, batch_size=64, random_sample=0, log_cte=jnp.log(15)):

  # Build the gradient function
  grad_U = jax.jit(jax.vmap(lambda x : -jax.grad(distributions['target'].log_prob)(x)))

  # Build the loglikelihood
  loglikelihood = jax.jit(lambda x : log_cte + distributions['target'].log_prob(x) - distributions['prior'].log_prob(x))

  # Build a function to evaluate a choice of parameters
  @partial(jax.vmap, in_axes=(0, 0, 0))
  def eval(t_params, neo_params, seed):
    # Build the transformation
    transformation = transformation_class(**t_params, grad_U=grad_U, dt=dt, Mdim=dim)
    neo_is = NEOIS(**neo_params,
          n_samples=n_samples,
          K=K,
          prior=distributions['prior'],
          importance_distr=distributions['importance_distr'],
          transformation=transformation)
    # Compute the normalizing constant
    return neo_is.estimate_E_T(loglikelihood, log_cte, seed)
  
  # Split a common dictionnary into two dictionnaries (for two sets of parameters)
  def split_dict(d):
    return { k : d[k] for k in transformation_ranges.keys() }, { k : d[k] for k in neo_ranges.keys() }
  
  # Multipky a seed
  def multiply_seed(initial_seed, times):
    return jnp.array([initial_seed] * times)
  
  # Build the list of parameters
  if random_sample > 0:
    it = ParameterSampler(dict(transformation_ranges, **neo_ranges), n_iter=random_sample, random_state=rng)
  else:
    it = ParameterGrid(dict(transformation_ranges, **neo_ranges))
  
  # Just compute it all
  results = []
  cur_seed = key
  for params in tqdm(dict_of_lists_to_list_of_dicts(list_of_dicts_to_dict_of_lists(it, batch_size))):
    transform_params, neo_params = split_dict(params)
    seeds = multiply_seed(cur_seed, len(transform_params['gamma']))
    log_z, log_E_T, _ = eval(transform_params, neo_params, seeds)
    results += dict_of_lists_to_list_of_dicts({
      'Z' : jnp.exp(log_z),
      'E_T' : jnp.exp(log_E_T),
      'neo_params' : dict_of_lists_to_list_of_dicts(neo_params),
      'transformation_params' : dict_of_lists_to_list_of_dicts(transform_params)
    })

  return results

# Get the best parameters
def get_best_params(r, true_cte=jnp.log(15)):
  return min(r, key=lambda x : jnp.abs(x['Z'] - jnp.exp(true_cte)))

# Classic parameter ranges
default_transformation_ranges = {
    'gamma' : jnp.linspace(0.01, 1.5, num=20),
    'Mscale' : jnp.linspace(0.001, 3.5, num=20)
}
default_neo_ranges = {
  'momentum_scale' : jnp.linspace(0.5, 3.5, num=20)
}