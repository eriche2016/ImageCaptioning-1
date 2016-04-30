
import json
import re
import nltk

TRAIN_FILE = 'data/annotations/captions_train2014.json'
VAL_FILE = 'data/annotations/captions_val2014.json'

regex = re.compile('[^a-zA-Z\s]')

fout = open('data/annotations/id2nouns.txt')

i = 0
for filename in [TRAIN_FILE, VAL_FILE]:
    data = json.load(open(filename))
    for anno in data['annotations']:
        if i % 10000 == 0:
            print 'tagging {}'.format(i)
        i += 1
        id, caption = anno['image_id'], anno['caption']:lower()
        caption = regex.sub('', caption)
        nouns = []
        for word, tag in nltk.pos_tag(caption.split()):
            if tag in ['NN', 'NNS']:
                nouns.append(word)
        fout.write("{} {}\n".format(id, " ".join(reversed(nouns))))
        if i % 10000 == 0:
            print " ".join(reversed(nouns))
fout.close()