import os
#import cmdstanpy
#cmdstanpy.install_cmdstan()
from cmdstanpy import cmdstan_path, CmdStanModel

run_model_1 = False
run_model_2 = False
run_model_3 = False #takes about 30 mins to run

if run_model_1 :
    stan = 'model1.stan'
    model = CmdStanModel(stan_file=stan)
    model.name
    model.stan_file
    model.exe_file
    model.code()

    data = 'data.json'
    variational_vb = model.variational(data=data, output_dir='.')

    a=[]
    for i in range(1,101):
        a.append(variational_vb.variational_params_dict['a['+str(i) + ']'])
    np.save('a.npy',a)

if run_model_2:
    !gdown https://drive.google.com/uc?id=115re9bhyhZBHHKsl7XV893fVN-Rpo1Y9&export=download
    stan = 'model2.stan'
    model = CmdStanModel(stan_file=stan)
    model.name
    model.stan_file
    model.exe_file
    model.code()

    data = 'data.json'
    variational_vb = model.variational(data=data, output_dir='.')

    dim=100
    a=[]
    s=[]
    for i in range(1,dim+1):
        a.append(variational_vb.variational_params_dict['a['+str(i) + ']'])
        s.append(variational_vb.variational_params_dict['s['+str(i) + ']'])

    a=np.array(a)
    s=np.array(s)

    np.save('a_2.npy',a)
    np.save('s_2.npy',s)

if run_model_3 :
    !gdown https://drive.google.com/uc?id=1wKW0VxtSU2f7kvrjWsMHoJqjwJlX4zZJ&export=download
    stan = 'model3.stan'
    model = CmdStanModel(stan_file=stan)
    model.name
    model.stan_file
    model.exe_file
    model.code()

    data = 'data.json'
    variational_vb = model.variational(data=data, output_dir='.')

    a = np.zeros((100,100))
    for i in range(1,101):
        for j in range(1,101):
            a[i-1,j-1]=variational_vb.variational_params_dict['a['+ str(i) + '][' + str(j) + ']']
    np.save('a_model3.npy',a)
