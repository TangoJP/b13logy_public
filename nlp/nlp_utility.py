from nltk.corpus import stopwords
import spacy

# list of stop words
stop_words = stopwords.words('english')

# Import spacy model
nlp = spacy.load('en_core_web_lg')

def lemmatize_text(text):
    doc = nlp(text)
    lemm_text = [
            token.lemma_.lower() for token in doc 
            if token.text != '\u2009'
            if token.is_alpha
            if not token.is_stop
            if not token.is_punct
            if not token.is_bracket
            if not token.is_quote
            if token.pos_ != 'PRON'
            if token.tag_ != 'BES'
            if token.tag_ != 'IN'
            if token.tag_ != 'HVS'
            if token.tag_ != 'PDT'
            if token.tag_ != 'TO'
            if token.tag_ != 'UH'
            if token.text not in stop_words
            if len(token.text) > 1
    ]
    return lemm_text