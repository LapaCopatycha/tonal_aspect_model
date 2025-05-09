import re
import torch

def prep_review(phrase, navec_emb):
    phrase = phrase.lower().replace('\ufeff', '').strip()
    phrase = re.sub(r'[^А-яA-z- ]', '', phrase)
    words = phrase.split()
    words = [torch.tensor(navec_emb[w]) for w in words if w in navec_emb]
    return words