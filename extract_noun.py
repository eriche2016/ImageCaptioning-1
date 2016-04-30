
import json
import re
import nltk

TRAIN_FILE = 'data/annotations/captions_train2014.json'
VAL_FILE = 'data/annotations/captions_val2014.json'

regex = re.compile('[^a-zA-Z\s]')

fout = open('data/annotations/id2nouns.txt', 'w')

i = 0
for filename in [TRAIN_FILE, VAL_FILE]:
    data = json.load(open(filename))
    for anno in data['annotations']:
        if i % 1000 == 0:
            print 'tagging {}'.format(i)
        id, caption, ann_id = anno['image_id'], anno['caption'].lower(), anno['id']
        caption = regex.sub('', caption)
        nouns = []
        for word, tag in nltk.pos_tag(caption.split()):
            if tag in ['NN', 'NNS']:
                nouns.append(word)
        fout.write("{} {} {}\n".format(ann_id, id, " ".join(reversed(nouns))))
        if i % 1000 == 0:
            print ann_id, id, " ".join(reversed(nouns))
        i += 1
fout.close()