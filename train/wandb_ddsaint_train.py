import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="ddsaint")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)

    parser.add_argument("--lamb", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=1)
    parser.add_argument("--hard", type=bool, default=True)

    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--diffusion_w", type=float, default=0.3)
    parser.add_argument("--beta_sche", type=str, default="exp")

    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=0)
   
    args = parser.parse_args()

    params = vars(args)
    main(params)
