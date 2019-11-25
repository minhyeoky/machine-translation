from src.model.metric import compute_bleu


def get_sentence(x, start='<start>', end='<end>', pad='<unk>'):
  """

  Args:
    x: list of words tokenized by preprocessor

  Returns:
    string without control tokens

  """
  ctl_tokens = [start, end, pad]
  ret = []
  for each in x:
    if each not in ctl_tokens:
      ret.append(each)
    elif each == end:
      break

  return ' '.join(ret)


def get_bleu_score(x, y):
  """

  Args:
    x: list of original sentences
    y: list of inferenced sentences

  Returns:
    mean value of bleu scores
  """
  scores = []
  for _x, _y in zip(x, y):
    if isinstance(_x, str):
      _x, _y = _x.split(), _y.split()
    reference_corpus = [[get_sentence(_x)]]
    translation_corpus = [get_sentence(_y)]
    bleu_elements = compute_bleu(reference_corpus,
                                 translation_corpus,
                                 max_order=4,
                                 smooth=False)
    bleu_score = bleu_elements[0]
    scores.append(bleu_score)
  return sum(scores) / len(scores)
