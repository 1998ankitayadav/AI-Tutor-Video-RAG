import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity 
import joblib



def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

# a = create_embedding(["cat sat on the mat", "harray totally destroy my mind"])
# # print(a)

jsons = os.listdir("newjsons")   #list all the json
my_dicts = []
item_id = 0
for json_file in jsons:
    with open(f"newjsons/{json_file}") as f:
        content = json.load(f)["chunks"]
    print(f"Creating Embeddings for {json_file}")   
    embeddings = create_embedding([c['text'] for c in content])
    

    embeddings = create_embedding([c['text'] for c in content])

    for i, chunk in enumerate(content):
        chunk['chunk_id'] = item_id
        chunk['embedding'] = embeddings[i]
        item_id += 1
        my_dicts.append(chunk)

        

df = pd.DataFrame.from_records(my_dicts)
#save this data frame
joblib.dump(df, 'embeddings.joblib')

print(df.head())
print(df.shape)
# print(df['start'].value_counts())
# print(df['text'].duplicated().sum())







