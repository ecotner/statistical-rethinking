import tqdm
import torch.tensor as tt
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
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
        
    def train(self, num_steps, lr=1e-2):
        pyro.clear_param_store()
        self.guide = AutoMultivariateNormal(self, init_loc_fn=init_to_mean)
        svi = SVI(self, guide=self.guide, optim=Adam({"lr": lr}), loss=Trace_ELBO())
        loss = []
        for _ in tqdm.notebook.tnrange(num_steps):
            loss.append(svi.step())
        return loss