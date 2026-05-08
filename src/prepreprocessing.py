import os
import json
import pandas as pd

# CHANGE THIS to the path where your 'train', 'dev', and 'test' folders are
BASE_PATH = './data/raw/RACE' 

def parse_race_folders(split_name):
    data = []
    split_path = os.path.join(BASE_PATH, split_name)
    
    for level in ['high', 'middle']:
        folder_path = os.path.join(split_path, level)
        if not os.path.exists(folder_path):
            continue
            
        print(f"Processing {split_name}/{level}...")
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    
                    article = content['article']
                    # Pull the exact ID from the JSON (e.g., "middle4305.txt")
                    base_id = content.get('id', filename) 
                    
                    for i in range(len(content['questions'])):
                        row = {
                            # Creates IDs like: middle4305.txt_0, middle4305.txt_1
                            'id': f"{base_id}_{i}", 
                            'article': article,
                            'question': content['questions'][i],
                            'A': content['options'][i][0],
                            'B': content['options'][i][1],
                            'C': content['options'][i][2],
                            'D': content['options'][i][3],
                            'answer': content['answers'][i]
                        }
                        data.append(row)
    return pd.DataFrame(data)

# 1. Process all splits
train_df = parse_race_folders('train')
val_df = parse_race_folders('dev')
test_df = parse_race_folders('test')

# 2. Ensure output directory exists
os.makedirs('data/raw', exist_ok=True)

# 3. Save to CSV
train_df.to_csv('data/raw/train.csv', index=False)
val_df.to_csv('data/raw/val.csv', index=False)
test_df.to_csv('data/raw/test.csv', index=False)

print("\nSuccess! Files saved in data/raw/")
print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}")