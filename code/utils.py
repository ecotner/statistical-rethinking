import pandas as pd
import numpy as np

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