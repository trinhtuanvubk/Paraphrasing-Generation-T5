from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("results/checkpoint-23000/", local_files_only=True)

model = AutoModelForSeq2SeqLM.from_pretrained("results/checkpoint-23000/", local_files_only=True).to(device)

# def paraphrase(
#     question,
#     num_beams=5,
#     num_beam_groups=5,
#     num_return_sequences=5,
#     repetition_penalty=10.0,
#     diversity_penalty=3.0,
#     no_repeat_ngram_size=2,
#     temperature=0.7,
#     max_length=128
# ):
#     input_ids = tokenizer(
#         f'paraphrase: {question}',
#         return_tensors="pt", padding="longest",
#         max_length=max_length,
#         truncation=True,
#     ).input_ids
    
#     outputs = model.generate(
#         input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
#         num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
#         num_beams=num_beams, num_beam_groups=num_beam_groups,
#         max_length=max_length, diversity_penalty=diversity_penalty
#     )

#     res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

#     return res


def get_response(input_text,num_return_sequences,num_beams):
    batch = tokenizer(['Paraphrasing this sentence: ' + input_text],truncation=True,padding='longest',max_length=100, return_tensors="pt").to(device)
    translated = model.generate(**batch,max_length=100,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

if __name__=="__main__":
    text = 'I am really love Vietnamese people and want to go here to meet my friend'
    # print(paraphrase(text))
    print(text)
    print(get_response(text,5, 5))