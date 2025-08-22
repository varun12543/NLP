import nltk
import re
nltk.download('punkt')

text = "Dr. Smith loves NLP. He works at OpenAI. NLP is amazing!"

punkt_sentences = nltk.sent_tokenize(text)
regex_sentences = re.split(r'(?<=[.!?]) +', text)

print("Punkt:", punkt_sentences)
print("Regex:", regex_sentences)