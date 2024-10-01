from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Load the pre-trained CodeBERT model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Example dataset with code diff only (no instruction)
dataset = [
    {
        "code_diff": """@@ -123,7 +123,7 @@ public void processRequest(HttpServletRequest request) {
                         String password = request.getParameter("password");
                         if (password.equals("admin123")) {
                         + if (password != null && password.equals("admin123")) { }""",
        "output": """CVE Description: The application does not properly handle null values in the password field...
                     Commit Message: Fix authentication bypass vulnerability by adding a null check for the password field."""
    },
    # Add more examples here
]

# Function to combine the instruction with the code diff
def combine_instruction_and_diff(instruction, code_diff):
    return f"{instruction}\nCode Diff:\n{code_diff}"

# Define the instruction separately
instruction = "Based on the following code diff, generate the corresponding CVE description and commit message."

# Function to tokenize the input (instruction + code diff)
def tokenize_function(examples):
    combined_input = [combine_instruction_and_diff(instruction, example['code_diff']) for example in examples]
    return tokenizer(combined_input, padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_inputs = tokenize_function(dataset)
tokenized_labels = tokenizer([example['output'] for example in dataset], truncation=True, padding=True)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Create the Trainer, including the model, training args, and dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_inputs,
    eval_dataset=tokenized_labels
)

# Start fine-tuning the model
trainer.train()



