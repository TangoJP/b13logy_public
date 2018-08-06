import pandas as pd
import re
import numpy
import pickle
import spacy
import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Import quantification methods table
print('Loading original DF ...')
with open('base_data/pride_quant_labeled.pickle', 'rb') as infile:
    df = pickle.load(infile)
print('Original DF loaded ...')

# list of stop words
stop_words = stopwords.words('english')

# Import spacy model
nlp = spacy.load('en_core_web_lg')

# Function for lemmatizing strings. 
# Here, I do not remove any word based on pos_tags unlike unigram features
def lemmatize_text(text):
    doc = nlp(text)
    lemm_text = [token.lemma_.lower() for token in doc]
    
    lemm_text = ' '.join(lemm_text)
    lemm_text = lemm_text.replace(' - ', '-')
    lemm_text = lemm_text.replace(' .', '.')
    lemm_text = lemm_text.replace(' ,', ',')
    lemm_text = lemm_text.replace('( ', '(')
    lemm_text = lemm_text.replace(' )', ')')
    lemm_text = lemm_text.replace(' / ', '/')
    
    lemm_text = lemm_text.replace('\u2009', '')    # This is a special case applicable to iloc[0]
    
    return lemm_text

# Create a new dataframe with lemmatized text
df2 = df[['silac', 'ms1_label_free', 'spectrum_counting', 'tmt', 'itraq', 'label_free']]

t0 = time.time()

print('Lemmatizing sample_protocol ...')
df2.loc[:, 'sample_protocol'] = df.loc[:, 'sample_protocol'].apply(lambda x: lemmatize_text(x))

t1 = time.time()
print('sample_protocol processed in %ds ...' % (t1 - t0))

print('Lemmatizing data_protocol ...')
df2.loc[:, 'data_protocol'] = df.loc[:, 'data_protocol'].apply(lambda x: lemmatize_text(x))

t2 = time.time()
print('data_protocol processed in %ds ...' % (t2 - t1))

print('Lemmatizing description ...')
df2.loc[:, 'description'] = df.loc[:, 'description'].apply(lambda x: lemmatize_text(x))

t3 = time.time()
print('description processed in %ds ...' % (t3 - t2))

print(df2.head(5))

with open('nlp16/df4ngrams.pickle', 'wb') as outfile:
    pickle.dump(df2, outfile)
print('New DF serialized and saved.')


### END
