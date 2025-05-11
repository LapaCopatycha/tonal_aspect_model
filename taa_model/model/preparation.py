import re
import torch

def prep_review(phrase):
    phrase = phrase.lower().replace('\ufeff', '').strip()
    phrase = re.sub(r'[^А-яA-z- ]', '', phrase)
    words = phrase.split()
    return words

def words_to_emb(words, navec_emb):
    return [torch.tensor(navec_emb[w]) for w in words if w in navec_emb]