import os
#import cmdstanpy
#cmdstanpy.install_cmdstan()
from cmdstanpy import cmdstan_path, CmdStanModel

stan = 'model2.stan'
model = CmdStanModel(stan_file=stan)
model.name
model.stan_file
model.exe_file
model.code()

data = 'data.json'
variational_vb = model.variational(data=data, output_dir='.',save_diagnostics=True)
