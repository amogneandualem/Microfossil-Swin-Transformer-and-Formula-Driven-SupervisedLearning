import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
# New imports for advanced features
from torchvision import transforms 
from torch.cuda.amp import autocast, GradScaler 

import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== OPTIMIZED CONFIG ====================
class Config:
    # Paths
    DATA_DIR = "/aifs/user/home/amogneandualem/Microfossil Classification/All_dataset"
    PRETRAINED_DIR = "/aifs/user/home/amogneandualem/Microfossil Classification/Pre traiened models"
    OUTPUT_DIR = "./swin_final_results_advanced" # Optimized output folder name
    
    # Model settings
    MODEL_NAME = "swin_base_patch4_window7_224"
    NUM_CLASSES = 32
    IMAGE_SIZE = 224
    
    # Pre-trained models 
    PRETRAINED_MODELS = {
        "imagenet": "imagenet_21k_swin_base.pth.tar",
        "exfractal": "exfractal_21k_swin_base.pth.tar", 
        "rcdb": "rcdb_21k_swin_base.pth.tar"
    }
    
    # Training settings
    BATCH_SIZE = 16     # Increased batch size for GPU/AMP
    EPOCHS = 50         
    LEARNING_RATE = 1e-4 
    WEIGHT_DECAY = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    
    # --- NEW: Replicates Setting ---
    REPLICATES = 3 # Run the full experiment this many times

# Create base output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# Set random seeds for reproducibility (ensures fixed data split)
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(Config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

# ==================== HELPER FUNCTION: SAVE SAMPLE IMAGES ====================
def save_sample_images(dataset, output_dir, num_samples=5):
    """Saves original, resized, and augmented versions of sample images."""
    print(f"\n🖼️ Saving {num_samples} sample images (Original vs. Resized/Augmented)...")
    sample_dir = os.path.join(output_dir, "sample_images")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Temporary transform for Resizing only
    resize_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    ])
    
    # Temporary transform for Augmentation (without normalization for visualization)
    aug_transform_visual = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.RandAugment(num_ops=2, magnitude=9), 
        transforms.ToTensor(), 
    ])
    
    # Select random indices for samples
    indices = torch.randperm(len(dataset))[:num_samples]
    
    for i, idx in enumerate(indices):
        img_path, label = dataset.samples[idx.item()]
        class_name = dataset.idx_to_class[label]
        
        try:
            original_img = Image.open(img_path).convert('RGB')
            
            # 1. Save Original Image (Pre-processing)
            original_img.save(os.path.join(sample_dir, f"{i+1}_A_Original_{class_name}.jpg"))
            
            # 2. Save Resized Image (After Resizing/Before Augmentation)
            resized_img = resize_transform(original_img)
            resized_img.save(os.path.join(sample_dir, f"{i+1}_B_Resized_{class_name}.jpg"))

            # 3. Save Augmented Image (After Augmentation/Before Normalization)
            augmented_tensor = aug_transform_visual(original_img)
            # Convert tensor back to PIL Image (Permute C, H, W to H, W, C and scale to 0-255)
            augmented_img_np = (augmented_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            augmented_img = Image.fromarray(augmented_img_np)
            augmented_img.save(os.path.join(sample_dir, f"{i+1}_C_Augmented_{class_name}.jpg"))
            
        except Exception as e:
            print(f"Warning: Could not save sample image {img_path}: {e}")
            
    print("✅ Sample images saved in swin_final_results_advanced/sample_images/")


# ==================== EFFICIENT DATA PIPELINE (with RandAugment) ====================
class EfficientMicrofossilDataset(Dataset):
    """ Custom Dataset integrated with torchvision transforms (RandAugment). """
    def __init__(self, data_dir, max_samples_per_class=500):
        self.data_dir = data_dir
        self.classes = sorted([d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            images = [img for img in os.listdir(class_dir) 
                      if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
            images = images[:max_samples_per_class]
            for img_name in images:
                self.samples.append((os.path.join(class_dir, img_name), 
                                     self.class_to_idx[class_name]))
        
        # --- RandAugment for Training (Includes Resize & Normalization) ---
        self.train_transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandAugment(num_ops=2, magnitude=9), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # --- Standard Transform for Validation/Testing (Includes Resize & Normalization) ---
        self.eval_transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
        self.is_train = False # Flag set by main() after splitting

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.is_train:
                image_tensor = self.train_transform(image)
            else:
                image_tensor = self.eval_transform(image)
            
            return image_tensor, label
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            dummy_image = torch.zeros(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            return dummy_image, label

# ==================== IMPROVED MODEL LOADING ====================
def load_model_safely(model_type="imagenet"):
    """ Loads Swin Transformer model (via timm) and safely applies pre-trained weights. """
    print(f"🔄 Loading {model_type} model...")
    
    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=False,
        num_classes=Config.NUM_CLASSES
    )
    
    model_path = os.path.join(Config.PRETRAINED_DIR, Config.PRETRAINED_MODELS[model_type])
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            new_state_dict = {}
            for k, v in state_dict.items():
                k_clean = k.replace('module.', '').replace('backbone.', '')
                new_state_dict[k_clean] = v
            
            model_dict = model.state_dict()
            pretrained_dict = {}
            loaded_params_count = 0

            for k, v in new_state_dict.items():
                if 'head.weight' in k or 'head.bias' in k:
                    continue
                
                if k in model_dict and model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
                    loaded_params_count += v.numel()
                
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            
            print(f"✅ Loaded {loaded_params_count:,} parameters from {model_type}")
            
        except Exception as e:
            print(f"⚠️ Could not load weights for {model_type}: {e}. Model will start from scratch/timm default.")
    
    return model.to(Config.DEVICE)

# ==================== MEMORY-EFFICIENT TRAINER ====================
class MemoryEfficientTrainer:
    """ Implements Layer-Wise LR, Grad Clipping, and Automatic Mixed Precision. """
    def __init__(self, model, train_loader, val_loader, test_loader, model_name, replicate_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model_name = model_name
        
        # Directory is now specific to the replicate
        self.model_dir = os.path.join(replicate_dir, model_name)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # --- AMP Scaler Initialization ---
        self.scaler = GradScaler()
        
        # --- Layer-Wise Learning Rate ---
        BASE_LR = Config.LEARNING_RATE * 0.1 
        HEAD_LR = Config.LEARNING_RATE * 1.0
        
        head_params = [p for n, p in model.named_parameters() if 'head' in n and p.requires_grad]
        base_params = [p for n, p in model.named_parameters() if 'head' not in n and p.requires_grad]

        self.optimizer = optim.AdamW(
            [
                {'params': base_params, 'lr': BASE_LR, 'weight_decay': Config.WEIGHT_DECAY},
                {'params': head_params, 'lr': HEAD_LR, 'weight_decay': Config.WEIGHT_DECAY},
            ], 
            lr=BASE_LR
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=Config.EPOCHS)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_accuracy = 0.0
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, (images, labels) in enumerate(pbar):
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            current_lr = self.optimizer.param_groups[1]["lr"] 
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{current_lr:.6f}'
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, epochs):
        print(f"\n🚀 Training {self.model_name}")
        print("=" * 50)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.1f}s")
            print(f"  Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}%")
            print(f"  Val:   Loss {val_loss:.4f}, Acc {val_acc:.2f}%")
            
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_accuracy': val_acc,
                }, os.path.join(self.model_dir, "best_model.pth"))
                print(f"  💾 New best model saved: {val_acc:.2f}%")
            
            if (epoch + 1) % 10 == 0:
                self.save_progress(epoch + 1)
        
        return self.best_accuracy
    
    def save_progress(self, epoch):
        """Save training progress plots."""
        print(f"  Saving progress curves to epoch_{epoch}.png...")
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title(f'{self.model_name.upper()} Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Val Accuracy')
        plt.title(f'{self.model_name.upper()} Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, f'training_curves_epoch_{epoch}.png'), 
                     dpi=150, bbox_inches='tight')
        plt.close()
    
    def evaluate_final(self, test_loader, idx_to_class):
        """Final evaluation on test set."""
        print(f"\n📊 Evaluating {self.model_name} on test set...")
        
        best_model_path = os.path.join(self.model_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=Config.DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded best model from epoch {checkpoint['epoch']+1} (Val Acc: {checkpoint['val_accuracy']:.2f}%)")
        else:
            print("  ⚠️ Best model checkpoint not found. Evaluating on last epoch's model.")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                
                with autocast(): 
                    outputs = self.model(images)
                    
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        
        class_names = [idx_to_class[i] for i in range(Config.NUM_CLASSES)]
        class_report = classification_report(all_labels, all_preds, 
                                             target_names=class_names,
                                             output_dict=True, zero_division=0)
        
        print(f"🎯 Test Accuracy: {accuracy:.4f}")
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(18, 16)) 
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{self.model_name.upper()} - Test Accuracy: {accuracy:.4f}', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'), 
                     dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'test_accuracy': accuracy,
            'best_val_accuracy': self.best_accuracy,
            'class_report': class_report
        }

# ==================== SUB-ROUTINE FOR ONE EXPERIMENT RUN ====================
def run_experiment(dataset, train_loader, val_loader, test_loader, replicate_dir, idx_to_class):
    """Runs the full training pipeline for all 3 pre-trained models within one replicate."""
    model_types = ["imagenet", "exfractal", "rcdb"]
    replicate_results = {}
    
    for model_type in model_types:
        try:
            print(f"\n{'='*60}")
            print(f"🎯 PROCESSING {model_type.upper()} PRE-TRAINING")
            print(f"{'='*60}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load the model and its pre-trained weights
            model = load_model_safely(model_type)
            
            # Initialize the trainer with the specific replicate directory
            trainer = MemoryEfficientTrainer(model, train_loader, val_loader, test_loader, model_type, replicate_dir)
            best_val_acc = trainer.train(Config.EPOCHS)
            
            eval_results = trainer.evaluate_final(test_loader, idx_to_class)
            
            replicate_results[model_type] = {
                'val_accuracy': best_val_acc,
                'test_accuracy': eval_results['test_accuracy'],
                'model_path': os.path.join(replicate_dir, model_type, "best_model.pth"),
                'class_report': eval_results['class_report']
            }
            
            print(f"✅ Completed {model_type}: Test Acc = {eval_results['test_accuracy']:.4f}")
            
        except Exception as e:
            print(f"❌ Error during processing of {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    return replicate_results

# ==================== MAIN TRAINING PIPELINE (Modified for Replicates) ====================
def main():
    print("🚀 SWIN TRANSFORMER - MEMORY EFFICIENT TRAINING (ADVANCED OPTIMIZED)")
    print("=" * 60)
    print(f"📁 Base Output Directory: {Config.OUTPUT_DIR}")
    print(f"🔄 Number of Replicates: {Config.REPLICATES}")
    print(f"💻 Device: {Config.DEVICE}")
    print(f"📦 Batch size: {Config.BATCH_SIZE}")
    print(f"📅 Epochs per experiment: {Config.EPOCHS}")
    print(f"🧠 Model: {Config.MODEL_NAME}")
    print("=" * 60)
    
    if Config.DEVICE.type == 'cpu':
        print("\n🚨 WARNING: Job is running on CPU. Check SLURM logs for 'Device: cuda' confirmation.")
    
    # Load dataset only once (data is the same across all replicates)
    print("\n📁 Loading dataset...")
    dataset = EfficientMicrofossilDataset(Config.DATA_DIR, max_samples_per_class=500)
    
    # Split dataset only once (fixed split for fair comparison across replicates)
    total_len = len(dataset)
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    # Set the augmentation flag for each split
    train_dataset.dataset.is_train = True
    val_dataset.dataset.is_train = False
    test_dataset.dataset.is_train = False
    
    # Create data loaders only once
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"✅ Dataset split: Train {len(train_dataset)}, Val {len(val_dataset)}, Test {len(test_dataset)}")
    
    # Save sample images (Original, Resized, Augmented) before starting the training loop
    save_sample_images(dataset, Config.OUTPUT_DIR) 

    # --- Main Replicate Loop ---
    all_replicate_results = {}
    idx_to_class = dataset.idx_to_class

    for replicate in range(1, Config.REPLICATES + 1):
        # 1. Setup Replicate Environment
        # The training variability will come from the data loader shuffling and internal GPU non-determinism.
        REPLICATE_OUTPUT_DIR = os.path.join(Config.OUTPUT_DIR, f"Replicate_{replicate}")
        os.makedirs(REPLICATE_OUTPUT_DIR, exist_ok=True)

        print(f"\n\n==================== STARTING REPLICATE {replicate}/{Config.REPLICATES} ====================")
        
        # 2. Run the experiment for all 3 model types
        replicate_results = run_experiment(dataset, train_loader, val_loader, test_loader, 
                                           REPLICATE_OUTPUT_DIR, idx_to_class)
        
        # 3. Store results
        all_replicate_results[f'Replicate_{replicate}'] = replicate_results

    # Final summary and results consolidation (across all replicates)
    if all_replicate_results:
        print(f"\n{'='*60}")
        print(f"🏆 FINAL RESULTS SUMMARY ACROSS {Config.REPLICATES} REPLICATES")
        print(f"{'='*60}")
        
        final_summary_data = []
        
        for model_type in ["imagenet", "exfractal", "rcdb"]:
            test_accs = [all_replicate_results[r][model_type]['test_accuracy'] 
                         for r in all_replicate_results if model_type in all_replicate_results[r]]
            val_accs = [all_replicate_results[r][model_type]['val_accuracy'] 
                        for r in all_replicate_results if model_type in all_replicate_results[r]]
            
            if test_accs:
                mean_test_acc = np.mean(test_accs)
                std_test_acc = np.std(test_accs)
                mean_val_acc = np.mean(val_accs)

                final_summary_data.append({
                    'Model': model_type,
                    'Mean Best Val Accuracy (%)': f"{mean_val_acc:.2f}",
                    'Mean Test Accuracy': f"{mean_test_acc:.4f}",
                    'Test Std Dev': f"±{std_test_acc:.4f}"
                })
                print(f"{model_type:12} - Mean Val: {mean_val_acc:6.2f}%, Mean Test: {mean_test_acc:.4f} (Std Dev: {std_test_acc:.4f})")
        
        if final_summary_data:
            df = pd.DataFrame(final_summary_data).set_index('Model')
            df.to_csv(os.path.join(Config.OUTPUT_DIR, "final_summary_all_replicates.csv"))
            
            best_entry = max(final_summary_data, key=lambda x: float(x['Mean Test Accuracy']))
            best_model_name = best_entry['Model']
            best_acc = float(best_entry['Mean Test Accuracy'])

            print(f"\n🎯 BEST MODEL (Mean): {best_model_name} with {best_acc:.4f} accuracy")
            
            published_benchmark = 0.863
            if best_acc >= 0.90:
                print("🎉 CONGRATULATIONS! ACHIEVED 90%+ MEAN ACCURACY! 🎉")
            elif best_acc > published_benchmark:
                print(f"✅ Better than original paper's best CNN result ({published_benchmark})!")
            else:
                print("⚠️ Result below the published benchmark.")

        print(f"\n📁 All results saved to: {Config.OUTPUT_DIR}")
        print("  - final_summary_all_replicates.csv: Comparative table of mean results across all replicates.")
        print("  - sample_images/: Sample images showing Original, Resized, and Augmented versions.")
        print("  - Replicate_X/: Individual folders for each run with model checkpoints and plots.")

if __name__ == "__main__":
    main()