
import json

data = json.load(open('data/annotations/instances_train2014.json'))
cat_set = set()
for anno in data['annotations']:
    cat_set.add(anno['category_id'])
print cat_set