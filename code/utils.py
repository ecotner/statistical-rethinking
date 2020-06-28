import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.moral import moral_graph
from networkx.algorithms.dag import ancestors
from networkx.algorithms.shortest_paths import has_path
from pyro.infer import Predictive
from pyro.infer.mcmc import NUTS, MCMC
from pyro import poutine
import torch
import torch.tensor as tt

### Sample summarization and interval calculation

def HPDI(samples, prob):
    """Calculates the Highest Posterior Density Interval (HPDI)
    
    Sorts all the samples, then with a fixed width window (in index space), 
    iterates through them all and caclulates the interval width, taking the 
    maximimum as it moves along. Probably only useful/correct for continuous 
    distributions or discrete distributions with a notion of ordering and a large 
    number of possible values.
    Arguments:
        samples (np.array): array of samples from a 1-dim posterior distribution
        prob (float): the probability mass of the desired interval
    Returns:
        Tuple[float, float]: the lower/upper bounds of the interval
    """
    samples = sorted(samples)
    N = len(samples)
    W = int(round(N*prob))
    min_interval = float('inf')
    bounds = [0, W]
    for i in range(N-W):
        interval = samples[i+W] - samples[i]
        if interval < min_interval:
            min_interval = interval
            bounds = [i, i+W]
    return samples[bounds[0]], samples[bounds[1]]


def precis(samples: dict, prob=0.89):
    """Computes some summary statistics of the given samples.
    
    Arguments:
        samples (Dict[str, np.array]): dictionary of samples, where the key
            is the name of the sample site, and the value is the collection
            of sample values
        prob (float): the probability mass of the symmetric credible interval
    Returns:
        pd.DataFrame: summary dataframe
    """
    p1, p2 = (1-prob)/2, 1-(1-prob)/2
    cols = ["mean","stddev",f"{100*p1:.1f}%",f"{100*p2:.1f}%"]
    df = pd.DataFrame(columns=cols, index=samples.keys())
    if isinstance(samples, pd.DataFrame):
        samples = {k: np.array(samples[k]) for k in samples.columns}
    elif not isinstance(samples, dict):
        raise TypeError("<samples> must be either dict or DataFrame")
    for k, v in samples.items():
        df.loc[k]["mean"] = v.mean()
        df.loc[k]["stddev"] = v.std()
        q1, q2 = np.quantile(v, [p1, p2])
        df.loc[k][f"{100*p1:.1f}%"] = q1
        df.loc[k][f"{100*p2:.1f}%"] = q2
    return df

### Causal inference tools

def independent(G, n1, n2, n3=None):
    """Computes whether n1 and n2 are independent given n3 on the DAG G
    
    Can find a decent exposition of the algorithm at http://web.mit.edu/jmn/www/6.034/d-separation.pdf
    """
    if n3 is None:
        n3 = set()
    elif isinstance(n3, (int, str)):
        n3 = set([n3])
    elif not isinstance(n3, set):
        n3 = set(n3)
    # Construct the ancestral graph of n1, n2, and n3
    a = ancestors(G, n1) | ancestors(G, n2) | {n1, n2} | n3
    G = G.subgraph(a)
    # Moralize the graph
    M = moral_graph(G)
    # Remove n3 (if applicable)
    M.remove_nodes_from(n3)
    # Check that path exists between n1 and n2
    return not has_path(M, n1, n2)

def conditional_independencies(G):
    """Finds all conditional independencies in the DAG G
    
    Only works when conditioning on a single node at a time
    """
    tuples = []
    for i1, n1 in enumerate(G.nodes):
        for i2, n2 in enumerate(G.nodes):
            if i1 >= i2:
                continue
            for n3 in G.nodes:
                try:
                    if independent(G, n1, n2, n3):
                        tuples.append((n1, n2, n3))
                except:
                    pass
    return tuples

def marginal_independencies(G):
    """Finds all marginal independencies in the DAG G
    """
    tuples = []
    for i1, n1 in enumerate(G.nodes):
        for i2, n2 in enumerate(G.nodes):
            if i1 >= i2:
                continue
            try:
                if independent(G, n1, n2, {}):
                    tuples.append((n1, n2, {}))
            except:
                pass
    return tuples

def sample_posterior(model, num_samples, sites=None, data=None):
    p = Predictive(
        model,
        guide=model.guide,
        num_samples=num_samples,
        return_sites=sites,
    )
    if data is None:
        p = p()
    else:
        p = p(data)
    return {k: v.detach().numpy() for k, v in p.items()}

def sample_prior(model, num_samples, sites=None):
    return {
        k: v.detach().numpy()
        for k, v in Predictive(
            model,
            {},
            return_sites=sites,
            num_samples=num_samples
        )().items()
    }

def plot_intervals(samples, p):
    for i, (k, s) in enumerate(samples.items()):
        mean = s.mean()
        hpdi = HPDI(s, p)
        plt.scatter([mean], [i], facecolor="none", edgecolor="black")
        plt.plot(hpdi, [i, i], color="C0")
        plt.axhline(i, color="grey", alpha=0.5, linestyle="--")
    plt.yticks(range(len(samples)), samples.keys(), fontsize=15)
    plt.axvline(0, color="black", alpha=0.5, linestyle="--")
    
    
def WAIC(model, x, y, out_var_nm, num_samples=100):
    p = torch.zeros((num_samples, len(y)))
    # Get log probability samples
    for i in range(num_samples):
        tr = poutine.trace(poutine.condition(model, data=model.guide())).get_trace(x)
        dist = tr.nodes[out_var_nm]["fn"]
        p[i] = dist.log_prob(y).detach()
    pmax = p.max(axis=0).values
    lppd = pmax + (p - pmax).exp().mean(axis=0).log() # numerically stable version
    penalty = p.var(axis=0)
    return -2*(lppd - penalty)


def format_data(df, categoricals=None):
    data = dict()
    if categoricals is None:
        categoricals = []
    for col in set(df.columns) - set(categoricals):
        data[col] = tt(df[col].values).double()
    for col in categoricals:
        data[col] = tt(df[col].values).long()
    return data


def train_nuts(model, data, num_warmup, num_samples, num_chains=1):
    kernel = NUTS(model, adapt_step_size=True, adapt_mass_matrix=True, jit_compile=True)
    engine = MCMC(kernel, num_samples, num_warmup, num_chains=num_chains)
    engine.run(data, training=True)
    return engine


def traceplot(s, num_chains):
    fig, axes = plt.subplots(nrows=len(s), figsize=(12, len(s)*num_chains))
    for (k, v), ax in zip(s.items(), axes):
        plt.sca(ax)
        for c in range(num_chains):
            plt.plot(v[c], linewidth=0.5)
        plt.ylabel(k)
    plt.xlabel("Sample index")
    return fig

def trankplot(s, num_chains):
    fig, axes = plt.subplots(nrows=len(s), figsize=(12, len(s)*num_chains))
    ranks = {k: np.argsort(v, axis=None).reshape(v.shape) for k, v in s.items()}
    num_samples = 1
    for p in list(s.values())[0].shape:
        num_samples *= p
    bins = np.linspace(0, num_samples, 30)
    for i, (ax, (k, v)) in enumerate(zip(axes, ranks.items())):
        for c in range(num_chains):
            ax.hist(v[c], bins=bins, histtype="step", linewidth=2, alpha=0.5)
        ax.set_xlim(left=0, right=num_samples)
        ax.set_yticks([])
        ax.set_ylabel(k)
    plt.xlabel("sample rank")
    return fig


def unnest_samples(samples: dict, group_by_chain=False, depth=1):
    """Unnests samples from multivariate distributions
    
    The general index structure of a sample tensor is
    [[chains,] samples [,idx1, idx2, ...]]. Sometimes the distribution is univariate
    and there are no additional indices. So we will always unnest from the right, but
    only if the tensor has rank of 3 or more (2 in the case of no grouping by chains).
    """
    _samples = samples.copy()
    for k, s in samples.items():
        n_idx = len(s.shape) - (group_by_chain + 1)
        if n_idx > 0:
            for i in range(s.shape[-n_idx]):
                _samples[f"{k}[{i}]"] = s[...,i]
            del _samples[k]
    if depth >= 2:
        _samples = unnest_samples(_samples, group_by_chain, depth-1)
    return _samples