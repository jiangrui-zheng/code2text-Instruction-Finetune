import transformers
import torch


model_id = "/workspace/instruct_finetune_llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

diff = """
diff --git a/modules/m_sasl.c b/modules/m_sasl.c
index 38c7baeb3..93c5a0412 100644
--- a/modules/m_sasl.c
+++ b/modules/m_sasl.c
@@ -91,6 +91,12 @@ m_authenticate(struct Client *client_p, struct Client *source_p,
 		return 0;
 	}
 
+	if (*parv[1] == ':' || strchr(parv[1], ' '))
+	{
+		exit_client(client_p, client_p, client_p, "Malformed AUTHENTICATE");
+		return 0;
+	}
+
 	saslserv_p = find_named_client(ConfigFileEntry.sasl_service);
 	if (saslserv_p == NULL || !IsService(saslserv_p))
 	{
"""

gt_tokens = "AUTH  ENT  IC  AT  parameter AUTH  ENT  IC  AT  parameterSAS  Lm  _  s  as  l "

messages = [
    {"role": "system", "content": "You are an AI assistant that explains code differences (diffs) in detail. Your task is to map each important variable of following list to its corresponding line number and explain the changes made in the code."},
    {"role": "user", "content": "Here is a code difference (diff) along with a set of important tokens from the CVE description. Please provide a detailed explanation for each variable mentioned in the set, including the line number, the changes made, and their significance. \nDiff:" + diff + "\nImportant variables: [" + gt_tokens + "]\n"},]

prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)
print("Generated Prompt:\n", prompt)
print("----------------------------------------------------------")
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=1024,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])