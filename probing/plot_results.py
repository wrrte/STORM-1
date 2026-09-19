import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results over steps")
    parser.add_argument("-csv_path", type=str, required=True, help="Path to the evaluation CSV file")
    parser.add_argument("-save_path", type=str, default="eval_plot.png", help="Path to save the plot image")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: {args.csv_path} does not exist.")
        return

    # Read CSV and strip spaces from column names just in case
    df = pd.read_csv(args.csv_path)
    df.columns = df.columns.str.strip()
    
    if "step" not in df.columns or "episode_avg_return" not in df.columns:
        print("Error: CSV must contain 'step' and 'episode_avg_return' columns.")
        return

    # Sort by step to ensure the plot lines connect correctly in order
    df = df.sort_values(by="step")

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["episode_avg_return"], marker='o', linestyle='-', color='b')
    plt.xlabel("Step")
    plt.ylabel("Episode Average Return")
    plt.title(f"Evaluation Score over Steps\n({os.path.basename(args.csv_path)})")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(args.save_path)
    print(f"Plot successfully saved to {args.save_path}")

if __name__ == "__main__":
    main()
