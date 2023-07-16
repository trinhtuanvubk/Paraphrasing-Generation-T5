# Paraphrasing with T5

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
--sentence "What are the best places to see in Vietnam?" \
--num_return_sequences 5 \
--max_length 100
```

- Flag:
	- `--max_length`: maxium length of `sentence`.