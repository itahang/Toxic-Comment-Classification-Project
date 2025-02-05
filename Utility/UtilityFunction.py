import streamlit as st

import jax 
from jax import random
from Model.NN import *
import pickle
from Model.Preprocess import paddedTokens
import numpy as np

@st.cache_data
def giveMeRandom(random_state:int,noOfRandomNumbers:int):
    """
    A function for generating Random numbers
    its gives jax random values
    """
    assert random_state>0 and noOfRandomNumbers>0
    key = random.PRNGKey(42)
    keys = random.split(key, 5)
    return keys
@st.cache_data
def giveparamater(path:str):
    """
    A utility Function of loading paramater
    Takes the path of the paramater as input
    and returns an pickled file of paramaters
    """
    params = 0
    with open(path, 'rb') as file:
        params = pickle.load(file)
    return params


@st.cache_data
def giveEM(path:str):
    """
    A utility function for loading Embedding Matrix
    """
    return jnp.load(path)

@st.cache_data
def giveMeOutputs(params,EmbeddingMatrix,user_input,models,Forward):
    """
    An function for returning Outputs
    """
    return np.array(np.array(jax.nn.sigmoid(Forward(params, EmbeddingMatrix, paddedTokens(user_input), models))).reshape(-1, 1) > np.array([0.4,0.01,0.5,0.3,0.1,0.1]).reshape(-1,1) ,dtype=np.float32)
