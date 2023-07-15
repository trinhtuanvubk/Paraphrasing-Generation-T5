
import argparse
import torch

def get_args():

    parser = argparse.ArgumentParser(description='Train Intent Classification Model')
    parser.add_argument('--scenario', type=str, default='train')
    parser.add_argument('--dataset_name', type=str, default="humarin/chatgpt-paraphrases", help='dataset from huggingface')
    parser.add_argument('--model_name', type=str, default="t5-small", help='Model name')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--padding', type=str, default='longest', help='Type of padding')
    parser.add_argument('--metric', type=str, default="rouge", help='Metric to evaluate')
    parser.add_argument('--output_dir', type=str, default="./results", help='Path to results dir')
    parser.add_argument('--checkpoints_path', type=str, default="./checkpoints/", help='checkpoints path')
    parser.add_argument('--sentence', type=str, default="to check my bill", help='sentence test for inference')


    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args