import os
import numpy as np
import pandas as pd
import random

os.chdir("C:/Users/Miyazaki/Desktop/Simulation")



def simulation_data_gen(prefix,
                        datapoints = 1024,
                        noise_amp=1,
                        noise_mean=0,
                        noise_std=1,
                        gauss_amp=1,
                        gauss_mean=0,
                        gauss_std=1,
                        iter_num = 10):
    os.makedirs("./simulation_data_{}".format(prefix), exist_ok=True)
    os.chdir("./simulation_data_{}".format(prefix))
    data = np.zeros(datapoints)
    label = [0]
    for i in range(iter_num):
        #make x axis
        x_axis = np.linspace(-1, 1, datapoints)
        noise_data = np.array([random.gauss(noise_mean, noise_std)
                               for i
                               in range(datapoints)])
        gauss_data = gauss_amp * np.exp(-(x_axis - gauss_mean)**2 / (gauss_std**2))
        tempdata = noise_data + gauss_data
        data = np.vstack([data, tempdata])
        label.append("{0}_{1}".format(prefix,i))
    data= data.T
    df = pd.DataFrame(data, columns=label)
    np.savetxt('./x_axis.csv', x_axis)
    df.to_csv("./data_{}.csv".format(prefix))
    return data

data = simulation_data_gen("amp0",
                        datapoints = 1024,
                        noise_amp=1,
                        noise_mean=0,
                        noise_std=1,
                        gauss_amp=0,
                        gauss_mean=0,
                        gauss_std=0.25,
                        iter_num = 10)