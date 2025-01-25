import os
import argparse
from sklearn.model_selection import train_test_split

def save_file_list(file_list, file_path):
    with open(file_path, 'w') as f:
        for file in file_list:
            f.write(file + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Test split for dataset")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save train and val file lists")
    parser.add_argument('--val_size', type=float, default=0.05, help="Proportion of data to be used for training (default: 0.8)")
    
    args = parser.parse_args()
    
    all_files = []
    for root, dirs, files in os.walk(args.dataset_dir):
        for file in files:
            if file.endswith('.wav'):
                rel_path = os.path.relpath(os.path.join(root, file), args.dataset_dir)
                all_files.append(rel_path)
    
    train_files, val_files = train_test_split(all_files, test_size=args.val_size, random_state=42)
    
    os.makedirs(args.output_dir, exist_ok=True)

    save_file_list(train_files, os.path.join(args.output_dir, 'train_files.txt'))
    save_file_list(val_files, os.path.join(args.output_dir, 'val_files.txt'))

    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
