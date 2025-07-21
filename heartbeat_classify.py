import os
import argparse
import torch
from exp.exp_classification import Exp_Classification


def run_model(model_name, root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='classification')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model', type=str, default=model_name)
    parser.add_argument('--model_id', type=str, default=f'{model_name}_heartbeat')
    parser.add_argument('--data', type=str, default='UEA')
    parser.add_argument('--root_path', type=str, default=root_path)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--c_out', type=int, default=2)
    parser.add_argument('--use_gpu', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args([])

    exp = Exp_Classification(args)
    setting = f"{model_name}_Heartbeat"
    print(f"\n===== Running {model_name} on Heartbeat =====")
    exp.train(setting)
    exp.test(setting)


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, 'Heartbeat')
    for model in ['TimesNet', 'Crossformer']:
        run_model(model, dataset_path)


if __name__ == '__main__':
    main()
