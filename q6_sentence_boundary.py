import nltk
import re
nltk.download('punkt')

text = "Dr. Arthur loves NLP. He works at Red gear.INC. What a nice day!"

punkt_sentences = nltk.sent_tokenize(text)
regex_sentences = re.split(r'(?<=[.!?]) +', text)

print("Punkt:", punkt_sentences)
print("Regex:", regex_sentences)