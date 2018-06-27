import pandas as pd
import numpy as np
import spacy

df = pd.read_csv('../data/scraped_pubmed_latest.csv', low_memory=False)
print(df.columns)
# nlp = spacy.load('en')
# doc = nlp('Hellow Workd')
# for token in doc:
#     print('"' + token.text + '"')