import functools
from . import Preprocessor, SentencePassager


def _lib():
  try:
    import spacy
  except ImportError as e:
    raise ImportError('"pip install spacy" required', e)

@functools.lru_cache
def _nlp(model):
  try:
    return spacy.load(model, disable=['ner', 'parser', 'tok2vec'])
  except OSError as e:
    raise RuntimeError(f'"python -m spacy download {model}" required', e)


clear_cache = _nlp.cache_clear


def preprocessor_factory(model, supports_stem=True, stops=None):
  def wrapped(stem=True, remove_punct=True, remove_stops=True):
    nlp = _nlp(model)
    if supports_stem and stem:
      stemmer = lambda t: t.lemma_
    else:
      stemmer = lambda t: t.norm_
    term_filter = lambda t: True
    if remove_stops:
      if stops is not None:
        def filter_stops(f):
          return lambda t: f(t) and not t.is_stop
      else:
        def filter_stops(f):
          return lambda t: f(t) and str(t) not in stops
      term_filter = filter_stops(term_filter)
    if remove_punct:
      def filter_punct(f):
        return lambda t: f(t) and not t.is_punct
      term_filter = filter_punct(term_filter)
    return Preprocessor(nlp, stemmer=stemmer, term_filter=term_filter)
  return wrapped


def passager_factory(model):
  def wrapped(length=4, stride=2, text_field='text', prepend_field='title'):
    nlp = _nlp(model)
    return SentencePassager(lambda x: list(str(s) for s in nlp(x).sents), length=length, stride=stride, text_field=text_field, prepend_field=prepend_field)
  return wrapped
