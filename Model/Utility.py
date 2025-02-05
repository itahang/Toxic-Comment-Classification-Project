import jax 
from jax import numpy as jnp
from functools import partial
from NN import LSTM
import optax

import jax
from jax import random
from Model.NN import *
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import trainers
import optax
import string

import spacy
import re
import emoji
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords



@jax.jit
def stable_softmax(z):
    z_stable = z - jnp.max(z)
    exp_z = jnp.exp(z_stable)
    return exp_z / jnp.sum(exp_z)

@partial(jax.jit,static_argnums=(2))
def ModelForward(params,x,models):
    *lstm,mlp = models
    h_0=x
    for i,model in enumerate(lstm):
        (_,_),h_0=model.FullforwardPass(params[i],h_0,model.c_0,model.h_0,LSTM.forward)
    return  (mlp.forward(params[-1],h_0[-1])).squeeze()

@partial(jax.jit,static_argnums=(1))
def loss_fn(params,models,x,y):
    Y=stable_softmax(ModelForward(params,x,models)).squeeze()
    return optax.safe_softmax_cross_entropy(Y,y)

def paddedTokens(text,length=30):
    txt= createTokens(text)
    txt = txt[:length]
    txt = jnp.array(txt).reshape(-1)
    if(len(txt)<length):
        diff = length-len(txt)
        txt = jax.lax.pad(txt,0,[(0, diff, 0)])
    return txt        

def createTokens(vocab,text):
    text= preprocess(text)
    ret=[]
    for word in text.split(' '):
        if word in  vocab:
            ret.append(vocab[word])
        else:
            ret.append(vocab['<unk>'])
    return ret


def create_embedding_matrix(vocab,glove_vectors, embedding_dim=100):
    vocab_size = len(vocab)
    weights = np.zeros((vocab_size, embedding_dim))
    
    for word, idx in vocab.items():
        if word in glove_vectors:
            weights[idx] = np.array(glove_vectors[word])
        elif word == "<pad>":
            weights[idx] = np.zeros(embedding_dim)  # Pad token
        else:
            # Initialize unknown words randomly
            weights[idx] = np.random.normal(embedding_dim) * 0.1
            
    return weights



def load_glove_model(glove_file_path):
    # Create an empty dictionary to store the word vectors
    glove_model = {}
    
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')  # Convert the rest to a numpy array
            glove_model[word] = vector
            
    return glove_model



nlp = spacy.load('en_core_web_sm')

def yield_tokens(data_iter):
    for text in data_iter:
        cleaned_text = preprocess(text)
        tokens = [
            token.lemma_ for token in nlp(cleaned_text)
            if 1 < len(token) < 25
        ]
        yield tokens


punc = string.punctuation
punc.replace('#', '')
punc.replace('!', '')
punc.replace('?', '')
punc = punc + "∞θ÷α•à−β∅³π‘₹´°£€\×™√²—"

chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don't care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can't stop laughing"
}


stpwds = stopwords.words('english')

nlp = spacy.load("en_core_web_sm")

time_zone_abbreviations = [
        "UTC", "GMT", "EST", "CST", "PST", "MST",
        "EDT", "CDT", "PDT", "MDT", "CET", "EET",
        "WET", "AEST", "ACST", "AWST", "HST",
        "AKST", "IST", "JST", "KST", "NZST"
    ]

patterns = [
    r'\\[nrtbfv\\]',         # \n, \t ..etc
    '<.*?>',                 # Html tags
    r'https?://\S+|www\.\S+',# Links
    r'\ufeff',               # BOM characters
    r'^[^a-zA-Z0-9]+$',      # Non-alphanumeric tokens
    r'ｗｗｗ．\S+',            # Full-width URLs
    r'[\uf700-\uf7ff]',      # Unicode private-use chars
    r'^[－—…]+$',            # Special punctuation-only tokens
    r'[︵︶]'                # CJK parentheses
]

def preprocess(text):
    for regex in patterns:
        text = re.sub(regex, '', text)
    
    text = text.lower()
    text = ' '.join(word for word in text.split() if word.upper() not in time_zone_abbreviations)
    text = ' '.join(chat_words.get(word.upper(), word) for word in text.split())
    text = ' '.join(word for word in text.split() if word.lower() not in stpwds)
    text = text.translate(str.maketrans(punc, ' ' * len(punc)))
    text = emoji.demojize(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text


# Initialize the tokenizer and trainer
tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
trainer = WordLevelTrainer(vocab_size=30002, special_tokens=["<pad>", "<unk>"])

# Set the pre-tokenizer
tokenizer.pre_tokenizer = Whitespace()


def Forward(params,x,models):
    lstm1,mlp1,mlp2 = models 
    (_,x),_=lstm1.FullforwardPass(params[0],x,lstm1.c_0,lstm1.h_0,lstm1.forward)
    x = jax.nn.relu( mlp1.forward(params[1],x))
    return (mlp2.forward(params[2],x))

Forward = jax.jit(Forward,static_argnums=2)