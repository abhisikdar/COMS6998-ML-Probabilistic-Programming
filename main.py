import os
#import cmdstanpy
#cmdstanpy.install_cmdstan()
from cmdstanpy import cmdstan_path, CmdStanModel

bernoulli_stan = '/Users/abhinavasikdar/Desktop/6998 ML w: PP/projecc/model1.stan'
bernoulli_model = CmdStanModel(stan_file=bernoulli_stan)
bernoulli_model.name
bernoulli_model.stan_file
bernoulli_model.exe_file
bernoulli_model.code()

bernoulli_data = '/Users/abhinavasikdar/Desktop/6998 ML w: PP/projecc/data.json'
bern_fit = bernoulli_model.sample(data=bernoulli_data, output_dir='.')