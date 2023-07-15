# Paraphrasing with T5

### Setup 

- To create Docker environment:
```
docker build -t text-env .
docker run -itd --gpus all --restart always -v $(pwd)/:/workspace --name text-env text-env:latest
docker exec -it text-env bash
```