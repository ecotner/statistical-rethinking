# Statistical Rethinking chapters

Each of the jupyter notebooks in this directory corresponds to a chapter of _Statistical Rethinking_. Yes, those are the actual names of the chapters, I did not come up with them!

## Chapter 0: [Preface](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/00_preface.ipynb)
Introduces the content of the book, how to use it effectively, some advice on coding, etc.

## Chapter 1: [The Golem of Prague](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/01_golem_of_prague.ipynb)
Discusses the zoo of common statistical tests, differences between hypotheses and models, some philosophy about truth/falsification of models. Introduces some of the fundamental differences between Frequentist and Bayesian statistics, then goes on to highlight specific future chapters: chapter 7 on model comparison, chapter 13 on multilevel models, chapters 5/6 on graphical causal models.

## Chapter 2: [Small Worlds and Large Worlds](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/02_small_large_worlds.ipynb)
Gives motivation for the Bayesian way of computing probabilities - it's just counting the ways observations could have occurred. Explains the various pieces of Bayes' rule: likelihood, prior, evidence, posterior. Illustrates how successive observations allow one to update their prior beliefs. Introduces the _grid approximation_ and _quadratic approximation_ (also known as _Laplace approximation_) for computing posteriors of simple models with low dimensionality and nearly Gaussian posteriors.

## Chapter 3: [Sampling the Imaginary](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/03_sampling_the_imaginary.ipynb)
Goes into detail about the grid approximation and how to compute posterior distributions from it. Discusses confidence and credible/compatibility intervals, HDI/HPDI/PI. Explains the benefits of having the entire posterior distribution over only having a point estimate. Shows how to sample from posteriors and how to use the samples to calculate any quantity of interest.

## Chapter 4: [Geocentric Models](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/04_geocentric_models.ipynb)
Explains why "common" models/distributions like linear regression and the Gaussian are the default in most scenarios and illustrates the Central Limit Theorem. Uses the grid/quadratic approximation to estimate the posterior of a simple linear regression model, and shows the importance of doing prior predictive sampling/simulation to determine logical/informative priors. Shows how to generalize to polynomial and spline regression models.

## Chapter 5: [The Many Variables & The Spurious Waffles](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/05_many_vars_and_spurious_waffles.ipynb)
This chapter starts looking at the correlation vs. causation problem, introduces some techniques for causal inference (DAG's), and how including certain predictors in your model could either increase/decrease bias if you're not careful. Shows how to use models to do counterfactual deduction. Explains how to incorporate categorical predictors into your linear models.

## Chapter 6: [The Haunted DAG & The Causal Terror](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/06_haunted_dag_and_causal_terror.ipynb)
Sources of bias and causality are discussed in more depth. Explains how multicollinearity can disguise causal relationships and introduce non-identifiability of parameters. Explains d-separation criteria and how to use it to make causal inferences when designing models. Illustrates Simpson's paradox, how to (try to) eliminate confounding, and create tests of causality.

## Chapter 7: [Ulysses' Compass](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/07_ulysses_compass.ipynb)
Introduces underfitting/overfitting problems, how to identify them through information criteria (AIC, WAIC, PSIS-LOO) and cross-validation techniques. Discusses various model fit metrics and when to use them (absolutely wrecks $R^2$ haha). Gives some more detail of the math of information theory underlying probability theory (entropy, KL divergence). Explains common pitfalls when comparing metrics of model fit. Shows how regularizing priors can be used to improve inference in the presence of domain knowledge or to reduce overfitting. Author heavily prefers using WAIC and PSIS-LOO over out-of-sample CV... not sure if I completely agree. Illustrates all this with a problem comparing models of primate brain mass.

## Chapter 8: [Conditional Manatees](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/08_conditional_manatees.ipynb)
Introduces nonlinear interactions between predictor variables, how to include them in "linear" models. Shows how rewriting more complicated models can eliminate identifiability issues. Shows how normalizing variables can make choosing priors simpler and more logical.

## Chapter 9: [Markov Chain Monte Carlo](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/09_mcmc.ipynb)
This chapter title is actually pretty apt. The concept of MCMC is introduced, various flavors (Metropolis, Gibbs sampling, Hamiltonian MC) are explained, and then we finally settle on HMC/NUTS, talk about some of the pros/cons, how to use it, how to diagnose problems.

## Chapter 10: [Big Entropy and the Generalized Linear Model](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/10_entropy_and_glm.ipynb)
The guiding principle of maximum entropy (to choose priors) is introduced. Genearlized linear models (GLM's), link functions, and techniques for interpreting parameters are introduced as well.

## Chapter 11: [God Spiked the Integers](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/11_god_spiked_ints.ipynb)
Integer-valued distributions like Poisson/Binomial are introduced, and GLM's are built using them. Brief discussion on how to account for censored data that comes up often in count models or survival analysis where you measure things like durations to an event, or events that cause "subjects" to "be removed from" the study.

## Chapter 12: [Monsters and Mixtures](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/12_monsters_and_mixtures.ipynb)
Mixture distributions are introduced, such as the zero-inflated Poisson, the beta-binomial, and gamma-poisson. Utility of these distributions and their higher entropy to help cover unexplained variance is discussed.

## Chapter 13: [Models with Memory](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/13_models_with_memory.ipynb)
Hierarchical/multilevel models are introduced. Advantages/disadvantages of pooling are discussed, and reparametrization (centered vs. non-centered) and its effects on HMC sampling efficiency are shown.

## Chapter 14: [Adventures in Covariance](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/14_adventures_in_covariance.ipynb)
Takes multilevel models further by introducing adaptive priors that can take covariance between groups of data into account (multidimensional Gaussians). Introduces Gaussian processes for groups linked by continuous values, which is illustrated in the context of geospatial and phylogenetic similarities.

## Chapter 15: [Missing Data and Other Opportunities](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/15_missing_data.ipynb)
Teaches how to deal with a variety of problems by modeling the data itself as distributions to be learned. It is shown how to treat measurement error and missing data as generative processes that allow the recovery of the "true" data. Lots of pitfalls related to the causal implications of using such techniques are discussed. Latent discrete variables and their treatment in HMC is also shown.

## Chapter 16: [Generalized Linear Madness](http://nbviewer.jupyter.org/github/ecotner/statistical-rethinking/blob/master/notebooks/16_generalized_linear_madness.ipynb)
Explains that while GLM's are a powerful tool, they are sometimes so general as to be uninterpretable. Often times, it is better to formulate a model using scientific theory inspired by the domain, trying to keep the model as close as possible to a plausible generative story. We go through several examples, including biological growth inspired by basic geometric principles, state space models for inferring strategies in children and forecasting population dynamics, highly nonlinear situations where we infer the parameters of a differential equations.

## Chapter 17: Horoscopes
This chapter doesn't have any code, so I did not make a notebook for it. It _very_ briefly discusses how statistical models should be used in scientific studies, and offers some guidelines on how scientific studies should be judged to improve the quality of scientific literature.