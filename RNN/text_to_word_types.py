import codecs
import nltk

with codecs.open('input_data/books_1_to_6.txt', "r", encoding='utf-8') as f:
    content = f.read()
tokenized_text = nltk.word_tokenize(content)
tagged_text = nltk.pos_tag(tokenized_text)

#print(tagged_text)

#for word, tag in tagged_text:
#    print(word + ": " + nltk.help.upenn_tagset(tag))

text_as_ids = []
tags = {}

for word, tag in tagged_text:
    if tag in tags:
        text_as_ids.append(tags[tag])
    else:
        tags[tag] = len(tags)
        text_as_ids.append(tags[tag])

print("Number of Word Types: "+str(len(tags)))
print("Length of processed text: "+str(len(text_as_ids)))