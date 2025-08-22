import nltk


for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)

text = "Natural Language Processing is an exciting field of AI."
tokens = nltk.word_tokenize(text)
tags = nltk.pos_tag(tokens)
print(tags)

