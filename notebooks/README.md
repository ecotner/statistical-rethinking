# Statistical Rethinking chapters

Each of the jupyter notebooks in this directory corresponds to a chapter of _Statistical Rethinking_. Yes, those are the actual names of the chapters, I did not come up with them!

## Chapter 0: Preface

## Chapter 1: The Golem of Prague

## Chapter 2: Small Worlds and Large Worlds

## Chapter 3: Sampling the Imaginary

## Chapter 4: Geocentric Models

## Chapter 5: The Many Variables & The Spurious Waffles
This chapter starts looking at the correlation vs. causation problem, introduces some techniques for causal inference (DAG's).

## Chapter 6: The Haunted DAG & The Causal Terror

## Chapter 7: Ulysses' Compass

## Chapter 8: Conditional Manatees

## Chapter 9: Markov Chain Monte Carlo
This chapter title is actually pretty apt. The concept of MCMC is introduced, various flavors (Gibbs sampling, Hamiltonian MC) are explained, and then we finally settle on HMC/NUTS, talk about some of the pros/cons, how to use it.

## Chapter 10: Big Entropy and the Generalized Linear Model
The guiding principle of maximum entropy (to choose priors) is introduced. Genearlized linear models (GLM's) are introduced as well.

## Chapter 11: God Spiked the Integers
Integer-valued distributions like Poisson/Binomial are introduced, and GLM's are built using them.

## Chapter 12: Monsters and Mixtures
Mixture distributions are introduced, such as the zero-inflated Poisson, the beta-binomial, and gamma-poisson. Utility of these distributions to help cover unexplained variance is discussed.

## Chapter 13: Models with Memory
Hierarchical/multilevel models are introduced. Advantages/disadvantages of pooling are discussed, and reparametrization (centered vs. non-centered) and its effects on HMC sampling efficiency are shown.

## Chapter 14: Adventures in Covariance
Takes multilevel models further by introducing adaptive priors that can take covariance between groups of data into account (multidimensional Gaussians). Introduces Gaussian processes for groups linked by continuous values, which is illustrated in the context of geospatial and phylogenetic similarities.