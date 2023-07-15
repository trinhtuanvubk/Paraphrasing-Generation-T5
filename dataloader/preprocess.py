import ast

def preprocess_function(args, examples, tokenizer):
    inputs = [f"Paraphrase this sentence: {doc}" for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=100, truncation=True)
    labels = [ast.literal_eval(i)[0] for i in examples['paraphrases']]
    labels = tokenizer(labels, max_length=100, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs