import tqdm
import torch.tensor as tt
import pyro
from pyro.infer import SVI, Trace_ELBO
import pyro.infer.autoguide
from pyro.infer.autoguide import AutoMultivariateNormal, AutoDiagonalNormal, init_to_mean,  AutoLaplaceApproximation
from pyro.optim import Adam

class RegressionBase:
    def __init__(self, df, categoricals=None):
        if categoricals is None:
            categoricals = []
        for col in set(df.columns) - set(categoricals):
            setattr(self, col, tt(df[col].values).double())
        for col in categoricals:
            setattr(self, col, tt(df[col].values).long())
            
    def __call__(self):
        raise NotImplementedError
        
    def train(self, num_steps, lr=1e-2, restart=True, autoguide=None, use_tqdm=True):
        if restart:
            pyro.clear_param_store()
            if autoguide is None:
                autoguide = AutoMultivariateNormal
            else:
                autoguide = getattr(pyro.infer.autoguide, autoguide)
            self.guide = autoguide(self, init_loc_fn=init_to_mean)
        svi = SVI(self, guide=self.guide, optim=Adam({"lr": lr}), loss=Trace_ELBO())
        loss = []
        if use_tqdm:
            iterator = tqdm.notebook.tnrange(num_steps)
        else:
            iterator = range(num_steps)
        for _ in iterator:
            loss.append(svi.step())
        return loss
