import re

def prep_review(phrase):
    phrase = phrase.lower().replace('\ufeff', '').strip()
    phrase = re.sub(r'[^А-яA-z- ]', '', phrase)
    return phrase