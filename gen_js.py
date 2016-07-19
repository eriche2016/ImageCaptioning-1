
import json

data = []
for i in range(1, 3):
    for line in open('server_test/captions{}.txt'.format(i)):
        inputs = line.strip().split('\t')
        data.append({'image_id': int(inputs[0]), 'caption': inputs[1]})
json.dump(data, open('server_test/captions_test2014_review_results.json', 'w'))