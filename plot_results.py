import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_model_data(model, base_dir):
    possible_names = ["training_metrics.csv", "traning_metrics.csv"]
    for name in possible_names:
        candidate = os.path.join(base_dir, model, name)
        if os.path.exists(candidate):
            return pd.read_csv(candidate)
    raise FileNotFoundError(
        f"CSV file not found for model {model} in directory {os.path.join(base_dir, model)}"
    )
def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
def save_or_show_plot(filename, save_dir, show_plot=False):
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved: {out_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()
def visualize_single_model(model, base_dir, save_dir, show_plot=False):
    try:
        df = load_model_data(model, base_dir)
    except Exception as e:
        print(e)
        return
    ensure_dir_exists(save_dir)
    # --- Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(df["Round"], df["Loss"], marker="o", color="red")
    plt.title(f"{model} - Loss per Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    save_or_show_plot(f"{model}_loss.png", save_dir, show_plot)
    # --- Total Reward ---
    plt.figure(figsize=(8, 5))
    plt.plot(df["Round"], df["Total Reward"], marker="o", color="blue")
    plt.title(f"{model} - Total Reward per Round")
    plt.xlabel("Round")
    plt.ylabel("Total Reward")
    plt.grid(True)
    save_or_show_plot(f"{model}_total_reward.png", save_dir, show_plot)

def compare_models(models, base_dir, save_dir, show_plot=False):
    data = {}
    for model in models:
        try:
            df = load_model_data(model, base_dir)
            data[model] = df
        except Exception as e:
            print(f"Error loading data for model {model}: {e}")
    if not data:
        print("No data available to display.")
        return
    ensure_dir_exists(save_dir)
    # --- Compare Loss ---
    plt.figure(figsize=(8, 5))
    for model, df in data.items():
        plt.plot(df["Round"], df["Loss"], marker="o", label=model)
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Compare Loss per Round among Models")
    plt.legend()
    plt.grid(True)
    save_or_show_plot("compare_loss.png", save_dir, show_plot)
    # --- Compare Total Reward ---
    plt.figure(figsize=(8, 5))
    for model, df in data.items():
        plt.plot(df["Round"], df["Total Reward"], marker="o", label=model)
    plt.xlabel("Round")
    plt.ylabel("Total Reward")
    plt.title("Compare Total Reward per Round among Models")
    plt.legend()
    plt.grid(True)
    save_or_show_plot("compare_total_reward.png", save_dir, show_plot)

def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--base_dir", type=str, default="tmp",
                               help="Directory containing model folders (default: tmp).")
    parent_parser.add_argument("--save_dir", type=str, default="results",
                               help="Directory to save plots (default: results).")
    parent_parser.add_argument("--show_plot", action="store_true",
                               help="Display plots (default: False, only save files).")
    parser = argparse.ArgumentParser(
        description="Visualize training metrics for one or more models."
    )
    subparsers = parser.add_subparsers(dest="command")
    parser_single = subparsers.add_parser(
        "single",
        parents=[parent_parser],
        help="Visualize metrics for a single model."
    )
    parser_single.add_argument("--model", type=str, required=True,
                               help="Folder name of the model to visualize.")
    parser_compare = subparsers.add_parser(
        "compare",
        parents=[parent_parser],
        help="Compare metrics among multiple models."
    )
    parser_compare.add_argument("--models", type=str, required=True,
                                help="List of models, separated by commas.")

    args = parser.parse_args()

    if args.command == "single":
        visualize_single_model(
            model=args.model,
            base_dir=args.base_dir,
            save_dir=args.save_dir,
            show_plot=args.show_plot
        )
    elif args.command == "compare":
        model_list = [m.strip() for m in args.models.split(",")]
        compare_models(
            models=model_list,
            base_dir=args.base_dir,
            save_dir=args.save_dir,
            show_plot=args.show_plot
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
