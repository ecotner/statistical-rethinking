import pandas as pd
import numpy as np
from networkx.algorithms.moral import moral_graph
from networkx.algorithms.dag import ancestors
from networkx.algorithms.shortest_paths import has_path

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