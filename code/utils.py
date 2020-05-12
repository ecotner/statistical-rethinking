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