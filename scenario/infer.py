from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def infer(args):

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoints_dir, local_files_only=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoints_dir, local_files_only=True).to(args.device)

    batch = tokenizer(['Paraphrasing this sentence: ' + args.sentence],
                    truncation=True,
                    padding='longest',
                    max_length=args.max_length,
                    return_tensors="pt").to(args.device)
    # print(model.generate.__dict__)
    translated = model.generate(**batch,
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_return_sequences,
                    temperature=1.5,
                    do_sample=True,
                    top_k=10)
    import inspect
    print(inspect.signature(model.generate))
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    print(tgt_text)
    return tgt_text

# if __name__=="__main__":
#     text = 'I am really love Vietnamese people and want to go here to meet my friend'
#     print(text)
#     # print(get_response(text,5, 5))
#     # print(infer)