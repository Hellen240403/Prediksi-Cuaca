import pandas as pd
import numpy as np
import random

def load_data(file):
    data = pd.read_csv(file)
    dataset  = data[['temp','hum','windspeed','weathersit']]
    return dataset

def train_model(data):
    temperature = np.array(data['temp'])
    Kelembapan = np.array(data['hum'])
    kecepatan_angin = np.array(data['windspeed'])
    Cuaca = np.array(data['weathersit'])

    learning_rate = 1
    bias = 1
    weights = [random.random(), random.random(), random.random(), random.random()]

    def Perceptron(input1, input2, input3, output):
        outputP = input1 * weights[0] + input2 * weights[1] + input3 * weights[2] + bias * weights[3]
        outputP = 1 / (1 + np.exp(-outputP))  # Sigmoid
        error = output - outputP
        weights[0] += error * input1 * learning_rate
        weights[1] += error * input2 * learning_rate
        weights[2] += error * input3 * learning_rate
        weights[3] += error * bias * learning_rate

    for _ in range(50):
        for i in range(len(temperature)):
            Perceptron(temperature[i], Kelembapan[i], kecepatan_angin[i], Cuaca[i])

    return weights

def predict(weights, x, y, z):
    bias = 1
    outputP = x * weights[0] + y * weights[1] + z * weights[2] + bias * weights[3]
    outputP = 1 / (1 + np.exp(-outputP))

    if outputP <= 1:
        return 'Cerah'
    elif outputP == 2:
        return 'Berawan'
    elif outputP == 3:
        return 'Gerimis'
    else:
        return 'Hujan'
