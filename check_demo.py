import numpy as np

def main():
    demo_path = '/media/storage_data/ai2lab/choemj/Drama_modified/demonstrations/Seaquest_20260729_225940.npz'
    print(f"Loading {demo_path}...")
    data = np.load(demo_path)
    
    print("obs shape:", data['obs'].shape, data['obs'].dtype)
    print("ram shape:", data['ram'].shape, data['ram'].dtype)
    
    divers = data['ram'][:, 62]
    print("\nDemonstration diver count distribution (RAM[62]):")
    for c in range(7):
        print(f"  count={c}: {(divers == c).sum()} frames")

if __name__ == "__main__":
    main()
