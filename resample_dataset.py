import os
import librosa
import soundfile as sf
import argparse

def resample_audio(input_dir, output_dir, target_sr):
    os.makedirs(output_dir, exist_ok=True)
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)
                
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                audio, sr = librosa.load(input_path, sr=None)
                audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                
                sf.write(output_path, audio_resampled, target_sr)
                print(f"Processed: {input_path} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample dataset")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save train and val file lists")
    parser.add_argument('--target_sr', type=int, required=True, help="Target sample rate of resampled dataset")
    args = parser.parse_args()
    resample_audio(args.dataset_dir, args.output_dir, args.target_sr)
