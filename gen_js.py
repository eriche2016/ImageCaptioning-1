
import json

# data = []
# for i in range(1, 3):
#     for line in open('server_test/captions{}.txt'.format(i)):
#         inputs = line.strip().split('\t')
#         data.append({'image_id': int(inputs[0]), 'caption': inputs[1]})
# json.dump(data, open('server_test/captions_test2014_review_results.json', 'w'))


from os import listdir
from os.path import isfile, join

data = []
for filename in listdir('data/val2014_features_googlenet'):
    if filename.endswith('dat'):
        id = int(filename.split('_')[-1].split('.')[0])
        data.append({'image_id': id, 'caption': ''})
json.dump(data, open('server_test/captions_val2014_review_results.json', 'w'))