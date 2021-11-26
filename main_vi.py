import os
#import cmdstanpy
#cmdstanpy.install_cmdstan()
from cmdstanpy import cmdstan_path, CmdStanModel

bernoulli_stan = 'model1.stan'
bernoulli_model = CmdStanModel(stan_file=bernoulli_stan)
bernoulli_model.name
bernoulli_model.stan_file
bernoulli_model.exe_file
bernoulli_model.code()

bernoulli_data = 'data.json'
# bern_fit = bernoulli_model.sample(data=bernoulli_data, output_dir='.')
bern_vb = bernoulli_model.variational(data=bernoulli_data, output_dir='.')
