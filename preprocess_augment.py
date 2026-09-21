from preprocess import save_augmentations
from pathlib import Path

if __name__ == "__main__":
    raw_data_dir = "Apple-Hyperspectral images"
    output_base_dir = "preprocessed_augmented_dataset"
    
    print(f"🔄 Starting offline augmentation...")
    print(f"   Input: {raw_data_dir}")
    print(f"   Output base: {output_base_dir}")
    print()
    
    cubes_dir, rgb_dir = save_augmentations(
        data_dir=raw_data_dir,
        out_dir=output_base_dir,
        img_size=224,
        augment_max=None
    )

    cubes_path = Path(cubes_dir)
    rgb_path = Path(rgb_dir)
    all_npy = list(cubes_path.rglob("*.npy"))
    all_pngs = list(rgb_path.rglob("*.png"))
    
    print()
    print(f"📊 Summary:")
    print(f"   Augmented hyperspectral cubes: {len(all_npy)}")
    print(f"   Preprocessed RGB images: {len(all_pngs)}")
    print()
    print(f"Next step: Train using the preprocessed RGB dataset")
    print(f"   python train_vgg_xgb.py --data-dir \"{rgb_dir}\" --out-dir \"trained_models\" --epochs 3 --batch-size 16")

