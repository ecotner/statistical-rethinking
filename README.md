# Statistical Rethinking

Going through the book _Statistical Rethinking_ (2nd edition) by Richard McElreath in an attempt to learn Bayesian modeling starting from zero. I'm a `python` kind of guy, so I think I'm going to try and redo all the code examples using one of the various PPL's (Probabilistic Programming Languages) that exist in the `python` universe. I have been getting more into `pytorch` lately as a framework for autodifferentiation and neural networks, and there is a nice-looking package called `pyro` for Bayesian inference that is built on top of it, so I will try and use that.

I think this is a much better idea than learning R so that I can copy McElreath's code, because I have learned so much more by implementing things from scratch rather than relying on his custom-built `quap`, `precis`, and other functions as black boxes that simply give you the answer and hide away a lot of the implementation details.

Du Phan, one of the maintainers of the package, is [doing something similar](https://fehiepsi.github.io/rethinking-pyro/), so their repo can serve as a comparison.

I will also use a mixture of `numpy`, `sklearn`, `pandas`, `matplotlib`, etc. for various other things if the need arises rather than go straight to `torch`/`pyro` (especially for simpler problems).

The data used can be found in the [official repository](https://github.com/rmcelreath/rethinking/tree/master/data) for the book. Noticed some files were missing (like `cars.csv`), but they can be found [here](https://github.com/fehiepsi/rethinking-numpyro/tree/master/data).