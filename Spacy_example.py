import spacy
nlp=spacy.load('en_core_web_sm')
doc=nlp('Tesla is ooking at buying U.S. startup Twitter for $6 million')
for token in doc:
    print(token.text,token.pos_)

print(nlp.pipeline)