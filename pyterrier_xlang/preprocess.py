from ast import Import
from cgitb import text
import string
from warnings import filters
import pyterrier as pt  
import stanza
from collections import Counter

class Preprocessor(pt.transformer.TransformerBase):
  def __init__(self, tokeniser, stemmer=None, preprocessor=None, text_fields=['title', 'text', 'body', 'query'], push_query=True, filters=None, return_toks=False):
    self.preprocessor = preprocessor
    self.tokeniser = tokeniser
    self.filters = filters
    self.stemmer = stemmer
    self.text_fields = [text_fields] if isinstance(text_fields, str) else text_fields
    self.push_query = push_query
    self.return_toks = return_toks

  def transform(self, df):
    if self.push_query and 'query' in df.columns:
      pt.model.push_queries(df)
    if hasattr(df, 'parallel_apply'):
      df = df.assign(**{f: df[f].parallel_apply(self.process_text) for f in self.text_fields if f in df.columns})
      if self.return_toks:
        df = df.assign(**{f + '_toks': df[f].parallel_apply(self.process_toks) for f in self.text_fields if f in df.columns})
    else:
      df = df.assign(**{f: df[f].apply(self.process_text) for f in self.text_fields if f in df.columns})
      if self.return_toks:
        df = df.assign(**{f + '_toks': df[f].apply(self.process_toks) for f in self.text_fields if f in df.columns})
    return df

  def process_toks(self, s):
    if self.preprocessor:
      s = self.preprocessor(s)
    toks = self.tokeniser(s)
    if self.filters:
      for f in self.filters:
        toks = f(toks)
      toks = list(filter(None, toks))
    if self.stemmer:
      toks = map(self.stemmer, toks)
    
    freqs = Counter(toks)
    return dict(freqs)

  def process_text(self, s):  
    if self.preprocessor:
      s = self.preprocessor(s)
    
    toks = self.tokeniser(s)

    if self.filters:
      for f in self.filters:
       toks = f(toks)
      toks = list(filter(None, toks)) #removes empty string
    

    if self.stemmer:
      toks = map(self.stemmer, toks)
    return ' '.join(toks)

class StanzaPreprocessor(pt.transformer.TransformerBase):
    def __init__(self, nlp, tokeniser, stemmer=None, preprocessor=None, term_filter=None, text_fields=['title', 'text', 'body', 'query'], push_query=True, filters=None):
      self.preprocessor = preprocessor
      self.nlp = nlp
      self.tokeniser = tokeniser
      self.term_filter = term_filter
      self.filters = filters
      self.stemmer = stemmer
      self.text_fields = [text_fields] if isinstance(text_fields, str) else text_fields
      self.push_query = push_query

    def transform(self, df):
      if self.push_query and 'query' in df.columns:
        pt.model.push_queries(df)
        df = df.assign(**{f: self.process_text(df[f]) for f in self.text_fields if f in df.columns})
      return df
    
    def process_text(self, column):
      in_docs = [stanza.Document([], text=text) for text in column]
      docs = self.nlp(in_docs)
      out_docs = self.tokeniser(docs)

      if self.filters:
        for f in self.filters:
          out_docs = list(map(f, out_docs))
      return out_docs # list of preproc text

def hazm_preprocessor(normalise=True, stem=True, remove_stops=True, remove_punct=True):
  '''
  Creates Preprocessor that uses hazm (Farsi only)
  '''
  try:
    import hazm
  except ImportError as e:
    raise ImportError("hazm module required to perform Farsi pre-processing please run 'pip install hazm'", e)
  
  stemmer = None
  
  if stem == 'lemma':
    stemmer = hazm.Lemmatizer().lemmatize
  elif stem:
    stemmer = hazm.Stemmer().stem

  filters = []

  if remove_stops:
    stopwords = set(hazm.stopwords_list())
    def filter_stops(toks):
      return [tok for tok in toks if tok not in stopwords]
    
    filters.append(filter_stops)

  if remove_punct:
    def filter_punct(toks):
      return [tok.translate(str.maketrans('', '', string.punctuation)) for tok in toks]
    
    filters.append(filter_punct)
  
  return Preprocessor(hazm.word_tokenize, stemmer=stemmer, preprocessor=hazm.Normalizer().normalize if normalise else None, filters=filters)

def spacy_preprocessor(model, supports_stem=True, remove_punct=True, remove_stops=True):
  '''
  Creates Preprocessor that uses spacy nlp model pipelines
  '''
  try:
    import spacy
  except ImportError as e:
      raise ImportError("Spacy module missing please run 'pip install spacy'", e)
  try:
    nlp = spacy.load(model, disable=['tok2vec', 'ner', 'tagger', 'parser'])
  except OSError as e:
      raise RuntimeError(f"Problem loading model {model} (you need to run 'python -m spacy download {model}' first)", e)
  
  if supports_stem:
    stemmer = lambda t: t.lemma_.lower()
  else:
    stemmer = lambda t: t.norm_
 
  filters = []

  if remove_stops:
    def filter_stops(toks):
      return [tok for tok in toks if not tok.is_stop]
    
    filters.append(filter_stops)
  
  if remove_punct:
    def filter_punct(toks):
      return [tok for tok in toks if not tok.is_punct]
    
    filters.append(filter_punct)

  return Preprocessor(nlp, stemmer=stemmer, filters=filters)

def spacy_tokeniser(remove_punct=True, remove_stops=True):
  '''
  Creates Preprocessor that uses spacy Tokenisers (currently only supports Farsi)
  '''
  try:
    from spacy.lang.fa import Persian
  except ImportError as e:
    raise ImportError("Spacy module required please run 'pip install spacy'", e)
  
  stemmer = lambda t: t.norm_
  
  filters = []
  
  if remove_stops:
    def filter_stops(toks):
      return [tok for tok in toks if not tok.is_stop]
    
    filters.append(filter_stops)

  if remove_punct:
    def filter_punct(toks):
      return [tok for tok in toks if not tok.is_punct]
    
    filters.append(filter_punct)

  return Preprocessor(tokeniser=Persian().tokenizer, stemmer=stemmer, filters=filters)

def snowball_preprocessor(lang, remove_punct=True, remove_stops=True):
  '''
  Creates Preprocessor that uses snowball
  '''
  try:
    from nltk.stem import SnowballStemmer
    from nltk.tokenize import word_tokenize  
    from nltk.corpus import stopwords
  except ImportError as e:
    raise ImportError("nltk module missing please run 'pip install nltk'", e)
  
  filters = []
  
  if remove_stops:
    stopwords = set(stopwords.words(lang))
    def filter_stops(toks):
      return [tok for tok in toks if tok not in stopwords]
    
    filters.append(filter_stops)

  if remove_punct:
    def filter_punct(toks):
      return [tok.translate(str.maketrans('', '', string.punctuation)) for tok in toks]
    
    filters.append(filter_punct)

  return Preprocessor(word_tokenize, stemmer=SnowballStemmer(lang).stem, filters=filters)

def jieba_preprocessor(remove_punct=True, remove_stops=True):
  '''
  Creates Preprocessor that uses jieba (Chinese only)
  '''
  try:
    import jieba
  except ImportError as e:
    raise ImportError("jieba module missing please run 'pip install jieba", e)
  try: 
    from stopwordsiso import stopwords
  except ImportError as e:
    raise ImportError("stopwordsiso module missing please run 'pip install stopwordsiso'",e)

  filters = []

  if remove_stops:
    stopwords = set(stopwords(['zh']))
    def filter_stops(toks):
      return [tok for tok in toks if tok not in stopwords]
    
    filters.append(filter_stops)
  
  if remove_punct:
    def filter_punct(toks):
      return [tok.translate(str.maketrans('', '', string.punctuation)) for tok in toks]
    
    filters.append(filter_punct)
  
  return Preprocessor(jieba.lcut, filters=filters)

def hgf_preprocessor(model, remove_punct=True):
  '''
  Creates Preprocessor that uses HuggingFace Tokenisers
  '''
  try:
    from transformers import  AutoTokenizer
  except ImportError as e:
    raise ImportError('Huggingface Transformers module missing, please run "pip install transformers')
  
  filters = []

  if remove_punct:
    def filter_punct(toks):
      return [tok.translate(str.maketrans('', '', string.punctuation)) for tok in toks]
    
    filters.append(filter_punct)

  tokenizer =  AutoTokenizer.from_pretrained(model)

  return Preprocessor(tokeniser=tokenizer.tokenize, filters=filters)  

def parsivar_preprocessor(normalise=True, stem=True, remove_punct=True):
  '''
  Creates Preprocessor that uses Parsivar (Farsi only)
  '''
  try:
    from parsivar import Normalizer, Tokenizer, FindStems
  except ImportError as e:
    raise ImportError('Parsivar required for preprocessing, please run "pip install parsivar"')
  
  filters = []

  if remove_punct:
    def filter_punct(toks):
      return [tok.translate(str.maketrans('', '', string.punctuation)) for tok in toks]
    
    filters.append(filter_punct)
  
  return Preprocessor(tokeniser=Tokenizer().tokenize_words, preprocessor=Normalizer().normalize if normalise else None, stemmer=FindStems().convert_to_stem if stem else None, filters=filters)

def ngram_preprocessor(N=3, char_level=True, remove_punct=True, filter_by_char=True):
  '''
  Creates Preprocessor that uses ntlk-based N-grams
  '''
  try:
    import nltk
    from nltk.util import ngrams
  except ImportError as e:
    raise ImportError('nltk required from preprocessing, please run "pip install nltk')
  
  filters = []

  if remove_punct:
    def filter_punct(toks):
      return [tok.translate(str.maketrans('', '', string.punctuation)) for tok in toks]
    
    filters.append(filter_punct)

  if char_level:
    def tokeniser(text, N=N):
      return ["".join(ngram) for ngram in ngrams(text,n=N)]
  else:
    def tokeniser(text, N=N):
      return ["".join(ngram) for ngram in ngrams(sequence=nltk.word_tokenize(text), n=N)]

  return Preprocessor(tokeniser=tokeniser, filters=filters)

def stanza_preprocessor(lang, stem=True, remove_punct=True):
  '''
  Creates Preprocessor that uses Stanza models
  '''
  try:
    import stanza
    import string
  except ImportError as e:
    raise ImportError("Stanza required for preprocessing, please run 'pip install stanza'", e)
  
  stanza.download(lang)
  
  filters = []

  if remove_punct:
    def filter_punct(text):
      return text.translate(str.maketrans('', '', string.punctuation))

    filters.append(filter_punct)

  if stem:
    processors = 'tokenize, lemma'
    def tokenize(docs):
      toks = []
      for entry in docs:
        entries = []
        for sentence in entry.sentences:
          for token in sentence.words:
            if token.lemma:
              entries.append(token.lemma)
            else:
              entries.append(token.text)
        toks.append(entries)
      return [' '.join(tok) for tok in toks]
  else:
    processors = 'tokenize'
    def tokenize(docs):
      toks = []
      for entry in docs:
        entries = []
        for sentence in entry.sentences:
          for token in sentence.words:
            entries.append(token.text)
        toks.append(entries)  
      return [' '.join(tok) for tok in toks]
    
  return StanzaPreprocessor(nlp=stanza.Pipeline(lang, processors=processors), tokeniser=tokenize, filters=filters)

def anserini_tokenizer(lang):
  '''
  Creates Preprocessor that uses Anserini Tokenisers
  '''
  try:
    from pyterrier_anserini import AnseriniTokenizer
  except ImportError as e:
    raise ImportError("Anserini Tokenizer required for preprocessing, please run 'pip install pyterrier-anserini'", e)

  if lang == 'nl':
    return Preprocessor(tokeniser=AnseriniTokenizer.nl.tokenize)
  elif lang == 'zh':
    return Preprocessor(tokeniser=AnseriniTokenizer.zh.tokenize, return_toks=True)
  elif lang == "id":
    return Preprocessor(tokeniser=AnseriniTokenizer.id.tokenize)
  elif lang == "fr":
    return Preprocessor(tokeniser=AnseriniTokenizer.fr.tokenize)
  elif lang == "it":
    return Preprocessor(tokeniser=AnseriniTokenizer.it.tokenize)
  elif lang == "en":
    return Preprocessor(tokeniser=AnseriniTokenizer.en.tokenize)
  else:
    raise ValueError(f"Anserini Tokenizer does not support {lang}")