import pandas as pd 
 
data = pd.read_csv('metadata.csv')

text = data['text']

with open('corpus.txt', 'w') as f:
    for i in text:
        line = f"{i}.\n"
        f.writelines(line)

print('done')