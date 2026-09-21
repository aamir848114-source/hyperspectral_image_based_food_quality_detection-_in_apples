from pathlib import Path
import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import xgboost as xgb
from preprocess import collect_files_labels, PreprocessedRGBDataset

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def print_and_save_results(model_name, X_train_feats, y_train_feats, X_train_feats, y_test_feats,
                           y_pred_train, y_pred_val, output_dir, num_files):
    train_acc = accuracy_score(y_train_feats, y_pred_train)
    train_precision = precision_score(y_train_feats, y_pred_train, average='macro', zero_division=0)
    train_recall = recall_score(y_train_feats, y_pred_train, average='macro', zero_division=0)
    train_f1 = f1_score(y_train_feats, y_pred_train, average='macro', zero_division=0)

    val_acc = accuracy_score(y_test_feats, y_pred_val)
    val_precision = precision_score(y_test_feats, y_pred_val, average='macro', zero_division=0)
    val_recall = recall_score(y_test_feats, y_pred_val, average='macro', zero_division=0)
    val_f1 = f1_score(y_test_feats, y_pred_val, average='macro', zero_division=0)
    
    print("\n" + "-"*80)
    print("TRAINING SET METRICS")
    print("-"*80)
    print(f"Accuracy: {train_acc:.4f} | Macro Precision: {train_precision:.4f} | Macro Recall: {train_recall:.4f} | Macro F1: {train_f1:.4f}")
    
    print("\n" + "-"*80)
    print("VALIDATION SET METRICS")
    print("-"*80)
    print(f"Accuracy: {val_acc:.4f} | Macro Precision: {val_precision:.4f} | Macro Recall: {val_recall:.4f} | Macro F1: {val_f1:.4f}")
    print("-"*80)

    summary_text = f"""VGG FEATURE EXTRACTION + {model_name.upper()} TRAINING REPORT
{'='*80}
DATASET: Total={num_files}, Train={len(X_train_feats)}, Val={len(X_train_feats)}, Features=25088
TRAIN SET: Acc={train_acc:.4f} | MacroPrec={train_precision:.4f} | MacroRec={train_recall:.4f} | MacroF1={train_f1:.4f}
VAL SET:   Acc={val_acc:.4f} | MacroPrec={val_precision:.4f} | MacroRec={val_recall:.4f} | MacroF1={val_f1:.4f}
{'='*80}
"""
    with open(output_dir / f"summary_{model_name}.txt", "w") as f:
        f.write(summary_text)
    
    print(f"✓ Results saved to {output_dir / f'summary_{model_name}.txt'}")
    
    return train_acc, train_precision, train_recall, train_f1, val_acc, val_precision, val_recall, val_f1

def save_feature_map_visualization(fmap, output_dir, prefix=""):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(8, 8, figsize=(14, 14))
    
    for idx in range(64):
        ax = axes[idx // 8, idx % 8]
        if idx < fmap.shape[0]:
            channel_map = fmap[idx]
            channel_map = (channel_map - channel_map.min()) / (channel_map.max() - channel_map.min() + 1e-8)
            ax.imshow(channel_map, cmap='plasma')
            ax.set_title(f'Ch {idx}', fontsize=8)
        ax.axis('off')
    
    fig.suptitle(f'{prefix} Feature Maps (first 64 channels)', fontsize=14)
    fig.tight_layout()
    filepath = output_dir / f"{prefix}feature_maps_grid.png"
    fig.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved feature maps grid to {filepath}")
    
    numpy_filepath = output_dir / f"{prefix}feature_maps.npy"
    np.save(numpy_filepath, fmap)
    print(f"✓ Saved raw feature maps (shape {fmap.shape}) to {numpy_filepath}")

def extract_conv_layer_activations(model, sample_input, device, output_dir, model_name=""):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conv_layers = [
        ('conv1', model.features[0]),
        ('conv2', model.features[5]),
        ('conv3', model.features[10]),
        ('conv4', model.features[19]),
    ]
    
    for layer_name, layer_module in conv_layers:
        layer_module.register_forward_hook(get_activation(layer_name))
    
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
    
    for layer_name, _ in conv_layers:
        if layer_name in activation:
            act = activation[layer_name].squeeze()
            
            if len(act.shape) == 3:
                num_channels = act.shape[0]
                grid_size = int(np.sqrt(min(num_channels, 64)))
                
                fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 14))
                axes = axes.flatten() if grid_size > 1 else [axes]
                
                for idx in range(min(num_channels, grid_size * grid_size)):
                    channel_map = act[idx].detach().cpu().numpy()
                    channel_map = (channel_map - channel_map.min()) / (channel_map.max() - channel_map.min() + 1e-8)
                    axes[idx].imshow(channel_map, cmap='plasma')
                    axes[idx].set_title(f'Ch {idx}', fontsize=8)
                    axes[idx].axis('off')

                for idx in range(min(num_channels, grid_size * grid_size), len(axes)):
                    axes[idx].axis('off')
                
                fig.suptitle(f'{model_name} - {layer_name} Activations (shape: {tuple(act.shape)})', fontsize=12)
                fig.tight_layout()
                
                filepath = output_dir / f"{model_name}_{layer_name}_activations.png"
                fig.savefig(filepath, dpi=100, bbox_inches='tight')
                plt.close(fig)
                print(f"✓ Saved {layer_name} activations to {filepath}")
    
    activation.clear()

def extract_features_from_vgg(model, loader, device, output_dir=None):
    model.eval()
    feat_extractor = nn.Sequential(model.features, model.avgpool)
    feat_extractor.to(device)
    
    feats = []
    labels = []
    saved_first_batch = False
    
    print("\nExtracting features from preprocessed images...")
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            xb = xb.to(device)
            
            fmap = feat_extractor(xb)
            
            if output_dir and not saved_first_batch:
                sample_fmap = fmap[0].cpu().numpy()
                save_feature_map_visualization(
                    sample_fmap, 
                    Path(output_dir) / "feature_maps", 
                    prefix="sample_apple_"
                )
                saved_first_batch = True
            
            out = fmap.view(fmap.size(0), -1)
            if out.size(1) != 512 * 7 * 7:
                print(f"Warning: feature vector size is {out.size(1)}, expected {512*7*7} (25088)")
            
            out = out.cpu().numpy()
            feats.append(out)
            labels.append(yb.numpy())
            
            total_samples += xb.size(0)
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {total_samples} images...")
    
    feats = np.vstack(feats)
    labels = np.concatenate(labels)
    
    print(f"✓ Feature extraction complete. Shape: {feats.shape}")
    return feats, labels

def train_and_evaluate_model(model_name, model_clf, X_train_feats, y_train_feats, X_train_feats, y_test_feats, vgg_model, sample_image, device, results_base, files, le):
    print(f"\n" + "="*80)
    print(f"STEP: {model_name.upper()} TRAINING ON EXTRACTED FEATURES")
    print(f"="*80)
    print(f"Training {model_name} classifier on {X_train_feats.shape[0]} training samples...")
    
    model_outdir = results_base / model_name
    model_outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {model_name} convolutional layer activations...")
    extract_conv_layer_activations(vgg_model, sample_image, device, model_outdir / "conv_activations", model_name=model_name)
    
    model_clf.fit(X_train_feats, y_train_feats)
    print(f"✓ {model_name} training complete")
    
    print(f"\n" + "="*80)
    print(f"{model_name.upper()} TRAIN-TEST SPLIT RESULTS")
    print("="*80)
    y_pred_train = model_clf.predict(X_train_feats)
    y_pred_val = model_clf.predict(X_train_feats)
    
    print_and_save_results(model_name, X_train_feats, y_train_feats, X_train_feats, y_test_feats,
                          y_pred_train, y_pred_val, model_outdir, len(files))
    
    print(f"\nSaving {model_name} outputs...")
    if model_name == "xgboost":
        model_clf.save_model(str(model_outdir / "xgb_model.json"))
        np.save(model_outdir / "X_train_feats.npy", X_train_feats)
        np.save(model_outdir / "y_train.npy", y_train_feats)
        np.save(model_outdir / "X_train_feats.npy", X_train_feats)
        np.save(model_outdir / "y_test.npy", y_test_feats)
        np.save(model_outdir / "classes.npy", le.classes_)
    else:
        with open(model_outdir / f"{model_name}_model.pkl", "wb") as f:
            pickle.dump(model_clf, f)

def main():
    data_dir = "preprocessed_augmented_dataset/augmented_preprocessed"
    img_size = 224
    batch_size = 32
    vgg_weights = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_base = Path("results") / timestamp
    results_base.mkdir(parents=True, exist_ok=True)
    
    print(f"="*80)
    print("VGG FEATURE EXTRACTION + MULTI-MODEL TRAINING")
    print(f"="*80)
    print(f"Data dir: {data_dir}")
    print(f"Results dir: {results_base}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    
    print(f"\nCollecting files...")
    files, labels = collect_files_labels(data_dir)
    if len(files) == 0:
        print("❌ No files found. Check data-dir and supported extensions.")
        return
    
    print(f"✓ Found {len(files)} files with labels")
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_labels, label_counts))}")

    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"\nPerforming train-test split (80-20)...")
    X_train, X_train, y_train, y_test = train_test_split(
        files, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"✓ Train: {len(X_train)}, Val: {len(X_train)}")

    print(f"\nCreating datasets...")
    train_ds = PreprocessedRGBDataset(X_train, list(y_train), img_size=img_size)
    val_ds = PreprocessedRGBDataset(X_train, list(y_test), img_size=img_size)

    print(f"Creating dataloaders...")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nLoading VGG16 pre-trained model...")
    device_obj = torch.device(device)
    vgg_model = torchvision.models.vgg16(pretrained=True)
    vgg_model = vgg_model.to(device_obj)
    print(f"✓ VGG16 loaded (pre-trained ImageNet weights)")

    if vgg_weights:
        print(f"Loading custom VGG weights from {vgg_weights}...")
        state_dict = torch.load(vgg_weights, map_location=device_obj)
        vgg_model.load_state_dict(state_dict, strict=False)
        print(f"✓ Custom weights loaded")

    print(f"\n" + "="*80)
    print("STEP 1: FEATURE EXTRACTION (VGG16 FORWARD PASS)")
    print(f"="*80)
    X_train_feats, y_train_feats = extract_features_from_vgg(vgg_model, train_loader, device_obj)
    X_train_feats, y_test_feats = extract_features_from_vgg(vgg_model, val_loader, device_obj)
    
    print(f"\nExtracting convolutional layer activations...")
    sample_image, _ = val_ds[0]
    sample_image = sample_image.unsqueeze(0).to(device_obj)

    clf_xgb = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric='mlogloss', verbosity=1,
        n_estimators=100, max_depth=3, learning_rate=0.05,
        reg_alpha=1.0, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8
    )
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1, verbose=1)
    
    train_and_evaluate_model("xgboost", clf_xgb, X_train_feats, y_train_feats, X_train_feats, y_test_feats,
                            vgg_model, sample_image, device_obj, results_base, files, le)
    train_and_evaluate_model("randomforest", clf_rf, X_train_feats, y_train_feats, X_train_feats, y_test_feats,
                            vgg_model, sample_image, device_obj, results_base, files, le)

    print(f"\n✓ All models saved to {results_base}")
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print(f"✓ Results: {results_base}")
    print("="*80)

if __name__ == "__main__":
    main()
