
import json
from collections import defaultdict as dd

filenames = ['data/annotations/instances_train2014.json', 'data/annotations/instances_val2014.json']

id2cats = dd(list)
for filename in filenames:
    data = json.load(open(filename))
    for anno in data['annotations']:
        id2cats[anno['image_id']].append(anno['category_id'])

fout = open('data/annotations/cats.parsed.txt', 'w')
for k, v in id2cats.iteritems():
    fout.write("{} {}\n".format(k, " ".join(map(str, v))))
fout.close()


# data = json.load(open('data/annotations/instances_train2014.json'))
# cat_set = set()
# for anno in data['annotations']:
#     cat_set.add(anno['category_id'])
# print cat_set