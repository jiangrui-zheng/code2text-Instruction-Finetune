from transformers import RobertaTokenizer, RobertaForMaskedLM, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
import re
import json


# Function to combine the instruction with the code diff
def combine_instruction_and_diff(instruction, code_diff):
    return f"{instruction}\nCode Diff:\n{code_diff}"

# Function to extract CVE identifier
def extract_cve(cve_string):
    match = re.match(r'(CVE-\d{4}-\d+)', cve_string)
    return match.group(1) if match else None

train_df = pd.read_json('/workspace/instruct_finetune_llama/output_Dataset_train.json')

# Transpose if necessary (as per your previous code)
train_df = train_df.transpose()

# Extract relevant columns for inputs (code diff) and labels (CVE description)
train_df['code_diff'] = train_df['diff_tokens'].apply(lambda x: ' '.join(x))  # Joining tokens for code diffs
train_df['cve_desc'] = train_df['cve_desc_tokens'].apply(lambda x: ' '.join(x))  # Joining tokens for CVE descriptions

# Convert the DataFrame to a Hugging Face Dataset
train_dataset_ = Dataset.from_pandas(train_df[['code_diff', 'cve_desc']])

train_dataset = train_dataset_.map(lambda example: {
    'instruction': "Based on the following code diff of "+ extract_cve(example['__index_level_0__']) + ", identify the line number where the variable described in the CVE appears. Provide only the line number as a digit.",
    'input': example['code_diff'],
    'output': example['cve_desc']
})
train_dataset = train_dataset.remove_columns(['__index_level_0__', 'code_diff', 'cve_desc'])
print(train_dataset[1])
dataset_as_list = [train_dataset[i] for i in range(len(train_dataset))]

with open('diff2text_train.json', 'w') as f:
    json.dump(dataset_as_list, f, indent=4)