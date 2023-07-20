# Paraphrasing with T5
This model generates and output sentence that preserves the meaning of input sentence with variations in word choise and grammar.

### Setup 

- To create Docker environment:
```
docker build -t paraphrasing-env .
docker run -itd --gpus all --restart always -v $(pwd)/:/workspace --name t5-env paraphrasing-env:latest
docker exec -it text-env bash
```

### Train

- To train:

```
python3 main.py \ 
--scenario train \
--num_epochs 30 \
--batch_size 16 \
--save_total_limit 3
```

- Flag:
	- `--save_total_limit`: number of times to save checkpoints.
### Inference

- To infer:

```
python3 main.py \
--scenario infer \
--sentence "What are the best place in Vietnam?" \
--num_return_sequences 5 \
--max_length 256
```

- Flag:
	- `--max_length`: maxium length of `sentence`.