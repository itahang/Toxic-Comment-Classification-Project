import streamlit as st
import jax 
from jax import random
from Model.NN import *
import pickle
from Model.Preprocess import paddedTokens
import numpy as np
import pandas as pd

@st.cache_data
def giveMeRandom():
    key = random.PRNGKey(42)
    keys = random.split(key, 5)
    return keys

keys = giveMeRandom()

lstm1 = LSTM(50, 32, keys[0])
mlp1 = MLP(32, 16, keys[1])
mlp2 = MLP(16, 6, keys[2])

params = 0

@st.cache_data
def giveparamater(paramaters):
    params = 0
    with open(paramaters, 'rb') as file:
        params = pickle.load(file)
    return params
params = giveparamater("fitting499.pkl")

models = (lstm1, mlp1, mlp2)

def giveForwardFunction():
    def Forward(params, Embedding, x, models):
        x = Embedding[x]
        lstm1, mlp1, mlp2 = models
        (_, x), _ = lstm1.FullforwardPass(params[0], x, lstm1.c_0, lstm1.h_0, lstm1.forward)
        x = jax.nn.relu(mlp1.forward(params[1], x))
        return (mlp2.forward(params[2], x))

    return jax.jit(Forward, static_argnums=3)

Forward = giveForwardFunction()

@st.cache_data
def giveEM(path):
    return jnp.load(path)

EmbeddingMatrix = giveEM("SavedModel/EmbeddingMatrix.npy")

def giveMeOutputs(user_input):
    return np.array(np.array(jax.nn.sigmoid(Forward(params, EmbeddingMatrix, paddedTokens(user_input), models))).reshape(-1, 1) > np.array([0.4,0.01,0.5,0.3,0.1,0.1]).reshape(-1,1) ,dtype=np.float32)

sigmoid_output= st.checkbox("Do you want to see model Sigmoid output?",True)


user_input = st.text_input("Enter Comment:")

   

if user_input != "":
    output = giveMeOutputs(user_input)
    if len(user_input.strip().split(' '))==1:
        output[:]=0.0

    if not sigmoid_output:
        output=jnp.asarray(output > jnp.array([0.5,0.01,0.1,0.01,0.01,0.2]).reshape(-1,1),dtype=jnp.float32)

    df = pd.DataFrame(output, columns=["Probability"])

    df["Category"] = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    st.bar_chart(df.set_index("Category"))
