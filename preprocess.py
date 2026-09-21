from pathlib import Path
import warnings
import cv2
import numpy as np
from tifffile import imread
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T

def find_label_from_path(p: Path):
    name = p.parts
    name_str = "/".join(name).lower()
    if "fresh" in name_str:
        return "fresh"
    if "high" in name_str:
        return "high"
    if "low" in name_str:
        return "low"
    return None

def load_hyper_image(path: Path):
    ext = path.suffix.lower()
    try:
        if ext in (".tif", ".tiff"):
            arr = imread(str(path))
            arr = np.asarray(arr)
            if arr.ndim == 3:
                return arr
            if arr.ndim == 2:
                return arr[:, :, None]
        elif ext == ".npy":
            return np.load(path)
        else:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Cannot read {path}")
            return img
    except Exception as e:
        warnings.warn(f"Failed to load {path}: {e}")
        return None

def collect_files_labels(data_dir):
    p = Path(data_dir)
    exts = (".tif", ".tiff", ".npy", ".jpg", ".jpeg", ".png")
    files = []
    labels = []
    for fp in p.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in exts:
            lbl = find_label_from_path(fp)
            if lbl is None:
                continue
            files.append(str(fp))
            labels.append(lbl)
    return files, labels

def to_rgb_from_cube(cube, target_size=None):
    if cube.ndim == 2:
        cube = cube[:, :, None]
    if cube.ndim == 3 and cube.shape[0] < cube.shape[2]:
        if cube.shape[2] > 3 and cube.shape[0] <= 3:
            cube = cube.transpose(1, 2, 0)
    H, W, B = cube.shape
    bands = cube.reshape(-1, cube.shape[2])
    try:
        pca = PCA(n_components=3)
        pcs = pca.fit_transform(bands)
        rgb = pcs.reshape(H, W, 3)
    except Exception:
        if cube.shape[2] >= 3:
            rgb = cube[:, :, :3]
        else:
            rgb = np.repeat(cube[:, :, :1], 3, axis=2)

    rgb = rgb.astype(np.float32)
    for c in range(3):
        ch = rgb[:, :, c]
        mn = ch.min()
        mx = ch.max()
        if mx - mn > 1e-6:
            rgb[:, :, c] = (ch - mn) / (mx - mn)
        else:
            rgb[:, :, c] = 0.0

    if target_size is not None:
        rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return rgb

def save_augmentations(data_dir, out_dir, img_size=224, augment_max=None):

    files, labels = collect_files_labels(data_dir)
    if len(files) == 0:
        raise RuntimeError("No files found to augment. Check data-dir and supported extensions.")

    max_images = augment_max if augment_max is not None else len(files)
    max_images = min(max_images, len(files))

    out_base = Path(out_dir)
    out_cubes_dir = out_base / "augmented_cubes"
    out_rgb_dir = out_base / "augmented_preprocessed"
    out_cubes_dir.mkdir(parents=True, exist_ok=True)
    out_rgb_dir.mkdir(parents=True, exist_ok=True)

    total_augmented = 0

    for i, fp in enumerate(files[:max_images]):
        p = Path(fp)
        rel = p.relative_to(Path(data_dir))
        cube = load_hyper_image(p)
        if cube is None:
            continue
        
        cube_variants = [
            ("orig", cube),
            ("hflip", np.fliplr(cube)),
            ("vflip", np.flipud(cube)),
            ("rot90", np.rot90(cube, k=1, axes=(0, 1))),
            ("rot180", np.rot90(cube, k=2, axes=(0, 1))),
            ("rot270", np.rot90(cube, k=3, axes=(0, 1))),
        ]
        
        if i >= 390:
            cube_variants = cube_variants[:5]

        dest_parent_cubes = out_cubes_dir / rel.parent
        dest_parent_rgb = out_rgb_dir / rel.parent
        dest_parent_cubes.mkdir(parents=True, exist_ok=True)
        dest_parent_rgb.mkdir(parents=True, exist_ok=True)
        stem = p.stem
        
        for suf, aug_cube in cube_variants:
            cube_name = dest_parent_cubes / f"{stem}_{suf}.npy"
            np.save(cube_name, aug_cube)
            
            rgb = to_rgb_from_cube(aug_cube, target_size=img_size)
            img_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            try:
                save_img = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
            except Exception:
                save_img = img_u8
            rgb_name = dest_parent_rgb / f"{stem}_{suf}.png"
            cv2.imwrite(str(rgb_name), save_img)
            
            total_augmented += 1

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{max_images} images ({total_augmented} augmented variants saved)")

    print(f"\n✅ Augmentation and preprocessing complete!")
    print(f"   Total augmented variants: {total_augmented}")
    print(f"   Augmented hyperspectral cubes: {out_cubes_dir}")
    print(f"   Preprocessed RGB images: {out_rgb_dir}")
    return out_cubes_dir, out_rgb_dir

class PreprocessedRGBDataset(Dataset):

    def __init__(self, files, labels, img_size=224):
        self.files = files
        self.labels = labels
        self.img_size = img_size
        self.imagenet_norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = Path(self.files[idx])
        
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        img = img.astype(np.float32) / 255.0
        
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        
        img_t = self.imagenet_norm(img_t)
        
        label = self.labels[idx]
        return img_t, label
