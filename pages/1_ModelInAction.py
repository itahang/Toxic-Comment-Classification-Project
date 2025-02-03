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
    keys = random.split(key,5)
    return keys

keys = giveMeRandom()

lstm1 = LSTM(50,32,keys[0])
mlp1 = MLP(32,16,keys[1])
mlp2 = MLP(16,6,keys[2])


params=0

@st.cache_data
def giveparamater(paramaters):
    params = 0
    with open(paramaters, 'rb') as file:
        params = pickle.load(file)
    return params
params = giveparamater("bestParams9.pkl")

models = (lstm1,mlp1,mlp2)



def giveForwardFunction():
    def Forward(params,Embedding,x,models):
        x= Embedding[x]
        lstm1,mlp1,mlp2 = models 
        (_,x),_=lstm1.FullforwardPass(params[0],x,lstm1.c_0,lstm1.h_0,lstm1.forward)
        x = jax.nn.relu( mlp1.forward(params[1],x))
        return (mlp2.forward(params[2],x))

    return jax.jit(Forward,static_argnums=3)

Forward = giveForwardFunction()


@st.cache_data
def giveEM(path):
    return jnp.load(path)

EmbeddingMatrix = giveEM("SavedModel/EmbeddingMatrix.npy")


def giveMeOutputs(user_input):
    return np.array(jax.nn.sigmoid(Forward(params,EmbeddingMatrix,paddedTokens(user_input),models))).reshape(-1,1)

user_input = st.text_input("Enter your name")
output= giveMeOutputs(user_input)

df = pd.DataFrame(output, columns=["Probability"])

# Define x-axis labels (ensure it's the same length as `output`)
df["Category"] = ["Toxic1","Toxic2","Toxic3","Toxic4","Toxic5","Toxic6"]

# Display the bar chart
if user_input is not "":
    st.bar_chart(df.set_index("Category"))
