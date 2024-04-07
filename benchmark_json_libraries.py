import json
import time

import orjson

with open("../twitter-50mb.json", "r", encoding="utf8") as f:
    content = f.read()

start = time.time()
json.loads(content)
print("Time taken for json.loads: ", time.time() - start)

start = time.time()
orjson.loads(content)
print("Time taken for orjson.loads: ", time.time() - start)
