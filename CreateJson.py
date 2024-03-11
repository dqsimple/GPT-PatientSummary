import json
import json
import tiktoken # for token counting
import numpy as np
from collections import defaultdict

encoding = tiktoken.get_encoding("cl100k_base")
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens
src_train_path = "data/NCA/src-train-summarized.txt"
tgt_train_path = "data/NCA/tgt-train-summarized.txt"
src_val_path = "data/NCA/src-val-summarized.txt"
tgt_val_path = "data/NCA/tgt-val-summarized.txt"
src_test_path = "data/NCA/src-test-summarized.txt"
tgt_test_path = "data/NCA/tgt-test-summarized.txt"

with open(src_train_path, 'r') as src_train, open(tgt_train_path, 'r') as tgt_train:
    json_data = []
    count = 0
    with open('data/NCA/NCA_train_small.jsonl', 'w', encoding='utf-8') as file:
        for src, tgt in zip(src_train, tgt_train):



            json_object = {
                "messages": [
                    {'role': 'system',
                     'content': "You are a highly skilled doctor with the ability to perfectly summarize a patient's health information. Note: '<PAR>' indicates a new line or paragraph separation. The requirements of the summary as below: 1. **Content**: Your summary should exclusively contain the patient's gender, age, race, current diagnoses, medications, and lab test results. 2. **Format**: Craft the information into a cohesive and informative paragraph. 3. **Accuracy**: Ensure that no required items are omitted. Do not include any past medical history or surgical information; focus solely on the present visit. 4. **Data Availability**: If specific information for the requested items does not exist, clearly indicate this by stating '<item name> data not found' without quotation marks. 5. **Relevance**: Avoid any mention of hospitalization-related matters or details outside the current visit's scope. Your goal is to provide a clear, concise, and accurate summary of the patient's current health status based on the presented visit's information. Remember to maintain a professional and empathetic tone throughout your summary."},
                    {'role': 'user', 'content': src.strip()},
                    {'role': 'assistant', 'content': tgt.strip()}
                ]
            }
            messages = json_object["messages"]
            #print(num_tokens_from_messages(messages))
            # 4096 or 16385
            if num_tokens_from_messages(messages) > 4096:
                print("TRAIN Over 4096")
            if tgt.strip:
                count += 1
            if count > 80:
                continue
            json_string = json.dumps(json_object, ensure_ascii=False)
            file.write(json_string + '\n')
            #json_data.extend(json_object)
            #json_data.append(json_object)

with open(src_val_path, 'r') as src_train, open(tgt_val_path, 'r') as tgt_train:
    json_data = []
    count = 0
    with open('data/NCA/NCA_val_small.jsonl', 'w', encoding='utf-8') as file:
        for src, tgt in zip(src_train, tgt_train):


            json_object = {
                "messages": [
                    {'role': 'system',
                     'content': "You are a highly skilled doctor with the ability to perfectly summarize a patient's health information. Note: '<PAR>' indicates a new line or paragraph separation. The requirements of the summary as below: 1. **Content**: Your summary should exclusively contain the patient's gender, age, race, current diagnoses, medications, and lab test results. 2. **Format**: Craft the information into a cohesive and informative paragraph. 3. **Accuracy**: Ensure that no required items are omitted. Do not include any past medical history or surgical information; focus solely on the present visit. 4. **Data Availability**: If specific information for the requested items does not exist, clearly indicate this by stating '<item name> data not found' without quotation marks. 5. **Relevance**: Avoid any mention of hospitalization-related matters or details outside the current visit's scope. Your goal is to provide a clear, concise, and accurate summary of the patient's current health status based on the presented visit's information. Remember to maintain a professional and empathetic tone throughout your summary."},
                    {'role': 'user', 'content': src.strip()},
                    {'role': 'assistant', 'content': tgt.strip()}
                ]
            }
            messages = json_object["messages"]
            #print(num_tokens_from_messages(messages))
            if num_tokens_from_messages(messages) > 4096:
                print("VAL Over 4096")
            if tgt.strip:
                count += 1
            if count > 10:
                continue
            json_string = json.dumps(json_object, ensure_ascii=False)
            file.write(json_string + '\n')


with open(src_test_path, 'r') as src_train, open(tgt_test_path, 'r') as tgt_train:
    json_data = []
    count = 0
    with open('data/NCA/NCA_test_small_message.jsonl', 'w', encoding='utf-8') as file1, \
            open('data/NCA/NCA_test_small_gt.jsonl', 'w', encoding='utf-8') as file2:
        for src, tgt in zip(src_train, tgt_train):
                #print("Missing value", count)
            json_object = {
                "messages": [
                    {'role': 'system',
                     'content': 'You are a doctor that can summary patient\'s health information perfectly.'},
                    {'role': 'user', 'content': src.strip()},
                    {'role': 'assistant', 'content': tgt.strip()}
                ]
            }

            messages = json_object["messages"]
            #print(num_tokens_from_messages(messages))
            if num_tokens_from_messages(messages) > 4096:
                print("Test Over 4096")
            if tgt.strip:
                count += 1
            json_object = [
                {'role': 'system',
                 'content': "You are a highly skilled doctor with the ability to perfectly summarize a patient's health information. Note: '<PAR>' indicates a new line or paragraph separation. The requirements of the summary as below: 1. **Content**: Your summary should exclusively contain the patient's gender, age, race, current diagnoses, medications, and lab test results. 2. **Format**: Craft the information into a cohesive and informative paragraph. 3. **Accuracy**: Ensure that no required items are omitted. Do not include any past medical history or surgical information; focus solely on the present visit. 4. **Data Availability**: If specific information for the requested items does not exist, clearly indicate this by stating '<item name> data not found' without quotation marks. 5. **Relevance**: Avoid any mention of hospitalization-related matters or details outside the current visit's scope. Your goal is to provide a clear, concise, and accurate summary of the patient's current health status based on the presented visit's information. Remember to maintain a professional and empathetic tone throughout your summary."},
                {'role': 'user', 'content': src.strip()}
            ]
            if count > 10:
                continue
            json_string1 = json.dumps(json_object, ensure_ascii=False)
            json_string2 = json.dumps(tgt.strip(), ensure_ascii=False)
            file1.write(json_string1 + '\n')
            file2.write(json_string2 + '\n')
