import streamlit as st

import jax 
from jax import random
from Model.NN import *
import pickle
from Model.Preprocess import paddedTokens
import numpy as np

@st.cache_data
def giveMeRandom(random_state,noOfRandomNumbers):
    assert random_state>0 and noOfRandomNumbers>0
    key = random.PRNGKey(42)
    keys = random.split(key, 5)
    return keys
@st.cache_data
def giveparamater(paramaters):
    params = 0
    with open(paramaters, 'rb') as file:
        params = pickle.load(file)
    return params


@st.cache_data
def giveEM(path):
    return jnp.load(path)

@st.cache_data
def giveMeOutputs(params,EmbeddingMatrix,user_input,models,Forward):
    return np.array(np.array(jax.nn.sigmoid(Forward(params, EmbeddingMatrix, paddedTokens(user_input), models))).reshape(-1, 1) > np.array([0.4,0.01,0.5,0.3,0.1,0.1]).reshape(-1,1) ,dtype=np.float32)
