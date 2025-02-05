import streamlit as st

import jax 
import pandas as pd

## Custom Modules
from Model.Preprocess import paddedTokens
from Model.NN import *
from Utility.UtilityFunction import *



keys = giveMeRandom(42,6)

lstm1 = LSTM(100,150,keys[0])
lstm2 = LSTM(150,200,keys[1])
mlp1 = MLP(200,100,keys[2])
mlp2 = MLP(100,64,keys[3])
mlp3 = MLP(64,32,keys[4])
mlp4 = MLP(32,6,keys[5])


params = giveparamater("fitting19.pkl")

models = (lstm1,lstm2,mlp1,mlp2,mlp3,mlp4)

@st.cache_data
def giveForwardFunction():
    def Forward(params,Embedding,x,models):
        x= Embedding[x]
        lstm1,lstm2,mlp1,mlp2 ,mlp3,mlp4= models 
        # return x
        (_,_),x=lstm1.FullforwardPass(params[0],x,lstm1.c_0,lstm1.h_0,lstm1.forward)
        # return x
        (_,x),_=lstm2.FullforwardPass(params[1],x,lstm2.c_0,lstm2.h_0,lstm2.forward)
        
        x = jax.nn.relu( mlp1.forward(params[2],x))
        x = jax.nn.relu( mlp2.forward(params[3],x))
        x = jax.nn.relu( mlp2.forward(params[4],x))
        return  mlp3.forward(params[5],x)

    return jax.jit(Forward, static_argnums=3)

Forward = giveForwardFunction()


EmbeddingMatrix = giveEM("SavedModel/Embedding 100d.npy")

def giveMeOutputs(user_input):
    pt = paddedTokens(user_input)
    return  jax.nn.sigmoid(Forward(params,EmbeddingMatrix,pt,models)).reshape(-1,1)

sigmoid_output= st.checkbox("Do you want to see model Sigmoid output?",True)    
user_input = st.text_input("Enter Comment:")


if user_input != "":
    output = giveMeOutputs(user_input)
    if len(user_input.strip().split(' '))==1:
        output[:]=0.0
    
    if not sigmoid_output:
        output=jnp.asarray(output > jnp.array([0.5,0.01,0.1,0.01,0.01,0.1]).reshape(-1,1),dtype=jnp.float32)

    df = pd.DataFrame(output, columns=["Probability"])
    df["Category"] = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    st.bar_chart(df.set_index("Category"))
