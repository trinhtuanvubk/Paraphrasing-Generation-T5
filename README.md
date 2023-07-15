# Paraphrasing with T5

### Setup 

- To create Docker environment:
```
docker build -t paraphrasing-env .
docker run -itd --gpus all --restart always -v $(pwd)/:/workspace --name t5-env paraphrasing-env:latest
docker exec -it text-env bash
```