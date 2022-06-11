from . import Preprocessor, SentencePassager
import string
import pyterrier as pt


def _lib():
  try:
    import camel_tools
    return camel_tools
  except ImportError as e:
    raise ImportError("pip install camel_tools required", e)


def preprocessor(normalise=True, stem=True, remove_stops=True, remove_punct=True):
  camel_tools = _lib()
  stemmer = None
  if stem == 'lemma':
    stemmer = hazm.Lemmatizer().lemmatize
  elif stem:
    stemmer = hazm.Stemmer().stem
  term_filter = None
  if remove_stops or remove_punct:
    term_filter = lambda t: True
    if remove_stops:
      stops = set(hazm.stopwords_list())
      def filter_stops(f):
        return lambda t: f(t) and t not in stops
      term_filter = filter_stops(term_filter)
    if remove_punct:
      def filter_punct(f):
        return lambda t: f(t) and t not in string.punctuation
      term_filter = filter_punct(term_filter)
  return Preprocessor(camel_tools.utils.normalize.normalize_unicode, stemmer=stemmer, preprocessor=hazm.Normalizer().normalize if normalise else None, term_filter=term_filter)


def passager(length=4, stride=2, text_field='text', prepend_field='title'):
  hazm = _lib()
  return SentencePassager(hazm.sent_tokenize, length=length, stride=stride, text_field=text_field, prepend_field=prepend_field)
