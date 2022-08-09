from ast import Import
from cgitb import text
import string
import pyterrier as pt  
import stanza

class Preprocessor(pt.transformer.TransformerBase):
  def __init__(self, tokeniser, stemmer=None, preprocessor=None, term_filter=None, text_fields=['title', 'text', 'body', 'query'], push_query=True, filter_by_char=False):
    self.preprocessor = preprocessor
    self.tokeniser = tokeniser
    self.term_filter = term_filter
    self.filter_by_char = filter_by_char
    self.stemmer = stemmer
    self.text_fields = [text_fields] if isinstance(text_fields, str) else text_fields
    self.push_query = push_query

  def transform(self, df):
    if self.push_query and 'query' in df.columns:
      pt.model.push_queries(df)
    if hasattr(df, 'parallel_apply'):
      df = df.assign(**{f: df[f].parallel_apply(self.process_text) for f in self.text_fields if f in df.columns})
    else:
      df = df.assign(**{f: df[f].apply(self.process_text) for f in self.text_fields if f in df.columns})
    return df

  def process_text(self, s):  
    if self.preprocessor:
      s = self.preprocessor(s)
    
    toks = self.tokeniser(s)
    if self.term_filter:
      if self.filter_by_char:
        toks = [''.join(filter(self.term_filter, list(tok))) for tok in toks]
      else:
        toks = filter(self.term_filter, toks)
    if self.stemmer:
      toks = map(self.stemmer, toks)

    return ' '.join(toks)

class StanzaPreprocessor(pt.transformer.TransformerBase):
    def __init__(self, nlp, tokeniser, stemmer=None, preprocessor=None, term_filter=None, text_fields=['title', 'text', 'body', 'query'], push_query=True):
      self.preprocessor = preprocessor
      self.nlp = nlp
      self.tokeniser = tokeniser
      self.term_filter = term_filter
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

      if self.term_filter:
        out_docs = list(map(self.term_filter, out_docs))
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
  term_filter = None
  if remove_stops or remove_punct:
    term_filter = lambda t: True
    if remove_stops:
      def filter_stops(f):
        return lambda t: f(t) and t not in set(hazm.stopwords_list())
      term_filter = filter_stops(term_filter)
    if remove_punct:
      def filter_punct(f):
        return lambda t: f(t) and t not in string.punctuation
      term_filter = filter_punct(term_filter)
  return Preprocessor(hazm.word_tokenize, stemmer=stemmer, preprocessor=hazm.Normalizer().normalize if normalise else None, term_filter=term_filter)


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
  term_filter = lambda t: True
  if remove_stops:
    def filter_stops(f):
      return lambda t: f(t) and not t.is_stop
    term_filter = filter_stops(term_filter)
  if remove_punct:
    def filter_punct(f):
      return lambda t: f(t) and not t.is_punct
    term_filter = filter_punct(term_filter)
  return Preprocessor(nlp, stemmer=stemmer, term_filter=term_filter)

def spacy_tokeniser(remove_punct=True, remove_stops=True):
  '''
  Creates Preprocessor that uses spacy Tokenisers (currently only supports Farsi)
  '''
  try:
    from spacy.lang.fa import Persian
  except ImportError as e:
    raise ImportError("Spacy module required please run 'pip install spacy'", e)
  stemmer = lambda t: t.norm_
  term_filter = lambda t: True
  if remove_stops:
    def filter_stops(f):
      return lambda t: f(t) and not t.is_stop
    term_filter = filter_stops(term_filter)
  if remove_punct:
    def filter_punct(f):
      return lambda t: f(t) and not t.is_punct
    term_filter = filter_punct(term_filter)
  return Preprocessor(tokeniser=Persian().tokenizer, stemmer=stemmer, term_filter=term_filter)

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
  term_filter = lambda t: True
  if remove_stops:
    def filter_stops(f):
      return lambda t: f(t) and t not in set(stopwords.words(lang))
    term_filter = filter_stops(term_filter)
  if remove_punct:
    def filter_punct(f):
      return lambda t: f(t) and t not in string.punctuation
    term_filter = filter_punct(term_filter)
  return Preprocessor(word_tokenize, stemmer=SnowballStemmer(lang).stem, term_filter=term_filter)

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
  term_filter = lambda t: True
  if remove_stops:
    def filter_stops(f):
      return lambda t: f(t) and t not in set(stopwords(['zh']))
    term_filter = filter_stops(term_filter)
  if remove_punct:
    def filter_punct(f):
      return lambda t: f(t) and t not in string.punctuation
    term_filter = filter_punct(term_filter)
  return Preprocessor(jieba.lcut, term_filter=term_filter)

def hgf_preprocessor(model, remove_punct=True):
  '''
  Creates Preprocessor that uses HuggingFace Tokenisers
  '''
  try:
    from transformers import  AutoTokenizer
  except ImportError as e:
    raise ImportError('Huggingface Transformers module missing, please run "pip install transformers')
  term_filter = None
  if remove_punct:
    term_filter = lambda t: True
    def filter_punct(f):
      return lambda t: f(t) and t not in string.punctuation
    term_filter = filter_punct(term_filter)

  tokenizer =  AutoTokenizer.from_pretrained(model)
  return Preprocessor(tokeniser=tokenizer.tokenize, term_filter=term_filter)  

def parsivar_preprocessor(normalise=True, stem=True, remove_punct=True):
  '''
  Creates Preprocessor that uses Parsivar (Farsi only)
  '''
  try:
    from parsivar import Normalizer, Tokenizer, FindStems
  except ImportError as e:
    raise ImportError('Parsivar required for preprocessing, please run "pip install parsivar"')
  
  term_filter = None
  if remove_punct:
    term_filter = lambda t: True
    def filter_punct(f):
      return lambda t: f(t) and t not in string.punctuation
    term_filter = filter_punct(term_filter)
  
  return Preprocessor(tokeniser=Tokenizer().tokenize_words, preprocessor=Normalizer().normalize if normalise else None, stemmer=FindStems().convert_to_stem if stem else None, term_filter=term_filter)

def ngram_preprocessor(N=3, char_level=True, remove_punct=True):
  '''
  Creates Preprocessor that uses ntlk-based N-grams
  '''
  try:
    import nltk
    from nltk.util import ngrams
  except ImportError as e:
    raise ImportError('nltk required from preprocessing, please run "pip install nltk')
  
  term_filter = None
  if remove_punct:
    term_filter = lambda c: True
    def filter_punct(f):
      return lambda c: f(c) and c not in string.punctuation
    term_filter = filter_punct(term_filter)
  
  if char_level:
    def tokeniser(text, N=N):
      return ["".join(ngram) for ngram in ngrams(text,n=N)]
  else:
    def tokeniser(text, N=N):
      return ["".join(ngram) for ngram in ngrams(sequence=nltk.word_tokenize(text), n=N)]

  return Preprocessor(tokeniser=tokeniser, term_filter=term_filter, filter_characters=True)

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

  if remove_punct:
    def filter_punct(text):
      return text.translate(str.maketrans('', '', string.punctuation))

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
    
  return StanzaPreprocessor(nlp=stanza.Pipeline(lang, processors=processors), tokeniser=tokenize, term_filter=filter_punct if remove_punct else None)
