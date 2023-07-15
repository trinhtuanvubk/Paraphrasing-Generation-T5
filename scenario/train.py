from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

from dataloader import data_loader
from .util import compute_metric_with_extra

class Trainer: 
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(args.device)
        self.tokenized_train, self.tokenized_val = data_loader(args, self.tokenizer)

        self.data_collator = DataCollatorForSeq2Seq(
                                    tokenizer=self.tokenizer, 
                                    model=self.model, 
                                    label_pad_token_id=-100, 
                                    padding=args.padding
                                    )
        
        self.compute_metrics = compute_metric_with_extra(self.args, self.tokenizer)

        self.training_args = Seq2SeqTrainingArguments(
                                    output_dir=args.output_dir,
                                    evaluation_strategy="epoch",
                                    learning_rate=args.learning_rate,
                                    per_device_train_batch_size=args.batch_size,
                                    per_device_eval_batch_size=args.batch_size,
                                    weight_decay=args.weight_decay,
                                    save_total_limit=args.save_total_limit,
                                    num_train_epochs=args.num_epochs,
                                    predict_with_generate=True
                                    )

    def fit(self):
        trainer = Seq2SeqTrainer(
                                model=self.model,
                                args=self.training_args,
                                train_dataset=self.tokenized_train,
                                eval_dataset=self.tokenized_val,
                                tokenizer=self.tokenizer,
                                data_collator=self.data_collator,
                                compute_metrics=self.compute_metrics
                                )
        trainer.train()

def train(args):
    trainer = Trainer(args)
    trainer.fit()
    self.model.save_pretrained("./results")


