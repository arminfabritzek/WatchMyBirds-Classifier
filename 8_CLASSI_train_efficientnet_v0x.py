

from PIL import Image
import os
import torch
from torchvision.transforms import AutoAugmentPolicy, AutoAugment
import albumentations as A
from albumentations.pytorch import ToTensorV2
torch.backends.cudnn.benchmark = True
import timm
from timm.data.mixup import Mixup
import platform
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import socket
import subprocess
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import itertools
from utils.info_to_json import InfoJSON
import utils.copy_training_data_for_github_classifier_v02 as copy_training_data_for_github_classifier_v02
from torch.amp import autocast  # GradScaler
import os
import shutil


# 1. Configuration & Setup
if platform.system() == "Darwin":
    BASE_DIRECTORY = "..."
elif platform.system() == "Linux":
    BASE_DIRECTORY = '...'
else:
    raise OSError("Unsupported operating system detected.")

# --- Input Data ---
# Point to the 'images' folder created by the data preparation script
dataset_input = os.path.join(BASE_DIRECTORY, "images")
if not os.path.exists(os.path.join(dataset_input, "train")):
     raise FileNotFoundError(f"Training data not found in {os.path.join(dataset_input, 'train')}. Please run the data preparation script first.")

# --- Output Directories ---
OUTPUT_BASE = os.path.join(BASE_DIRECTORY, "output_finetune")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT = os.path.join(OUTPUT_BASE, timestamp)
os.makedirs(OUTPUT, exist_ok=True)
LOG_DIR = os.path.join(OUTPUT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# --- Logging & Workers ---
file_path_current_run = os.path.join(OUTPUT, "training_settings_info_current_run.json")
info_manager_current_run = InfoJSON(file_path_current_run)
writer = SummaryWriter(log_dir=LOG_DIR)
cpu_count = os.cpu_count() or 2
_num_workers = max(cpu_count - 1, 1)

# --- Model & Training Hyperparameters ---
model_name = 'efficientnet_b2' # Example model


# Model-specific adjustments
if model_name == 'efficientnet_b0':
    # General Hyperparameters (can be overridden by model-specific section if needed)
    _batch_size             = 256
    _num_epochs             = 200
    _early_stop_patience    = 15
    _warmup_epochs          = 5
    _learning_rate          = 0.001
    _finetune_lr            = _learning_rate/10
    _adamW_weight_decay     = 5e-4
    _unfreeze_epoch         = _warmup_epochs + 5
    _initial_lr = _learning_rate / 100
elif model_name == 'efficientnet_b1':
    _batch_size             = 256
    _num_epochs             = 200
    _early_stop_patience    = 15
    _warmup_epochs          = 5
    _learning_rate          = 0.001
    _finetune_lr            = _learning_rate/10
    _adamW_weight_decay     = 6e-4
    _unfreeze_epoch         = _warmup_epochs + 5
    _initial_lr             = _learning_rate / 100
elif model_name == 'efficientnet_b2':
    _batch_size = 256
    _num_epochs = 200
    _early_stop_patience = 15
    _warmup_epochs = 5
    _learning_rate      = 0.001
    _finetune_lr =  _learning_rate/10
    _adamW_weight_decay = 7e-4
    _unfreeze_epoch = _warmup_epochs + 5
    _initial_lr = _learning_rate / 100
else:
    raise ValueError(f"Unsupported model: {model_name}")

print(f"--- Hyperparameters for {model_name} ---")
print(f"Batch Size: {_batch_size}")
print(f"Epochs: {_num_epochs} (Unfreeze at epoch {_unfreeze_epoch})")
print(f"Warmup Epochs: {_warmup_epochs}")
print(f"Initial Head LR: {_learning_rate}")
print(f"Full Finetune LR: {_finetune_lr}")
print(f"Warmup Start LR: {_initial_lr}")
print(f"Weight Decay: {_adamW_weight_decay}")
print(f"Early Stopping Patience: {_early_stop_patience}")
print(f"------------------------------------")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# TensorBoard Management
def is_tensorboard_running(host="localhost", port=6006, timeout=1):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.close()
        return True
    except socket.error:
        return False

def kill_tensorboard(port=6006):
    try:
        if platform.system() == "Windows":
             # Find PID using netstat and taskkill
             result = subprocess.run(['netstat', '-ano', '|', 'findstr', f':{port}'], capture_output=True, text=True, shell=True)
             lines = result.stdout.strip().split('\n')
             pids_to_kill = set()
             for line in lines:
                  if f'TCP' in line and f'0.0.0.0:{port}' in line:
                       parts = line.split()
                       pid = parts[-1]
                       pids_to_kill.add(pid)
             if pids_to_kill:
                  print(f"Found TensorBoard on port {port}, killing PIDs: {pids_to_kill}")
                  for pid in pids_to_kill:
                       subprocess.run(['taskkill', '/F', '/PID', pid])
                  time.sleep(2)

        else: # Linux/macOS
             proc = subprocess.check_output(["lsof", "-ti", f":{port}"])
             pids = proc.decode().split()
             if pids:
                 print(f"Found TensorBoard on port {port}, killing: {pids}")
                 for pid in pids:
                     subprocess.run(["kill", "-9", pid])
                 time.sleep(2)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("No TensorBoard process found or 'lsof'/'netstat' not available.")
    except Exception as e:
        print(f"Error killing TensorBoard: {e}")


if is_tensorboard_running():
    kill_tensorboard()

tb_command = f"tensorboard --logdir {LOG_DIR} --host 0.0.0.0 --port 6006"
try:
    creationflags = subprocess.DETACHED_PROCESS if platform.system() == "Windows" else 0
    tb_process = subprocess.Popen(tb_command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags)
    print("Attempting to start TensorBoard...")
    time.sleep(5) # Give it some time to start
    if is_tensorboard_running():
         print("TensorBoard started successfully.")
    else:
         print("TensorBoard may not have started correctly. Check manually.")
except Exception as e:
    print(f"Failed to start TensorBoard: {e}")


# Helper function
def copy_files(destination_mapping):
    """
    Copies files from their source locations to the corresponding destination folders.

    Parameters
    ----------
    destination_mapping : dict
        A dictionary where the key is the full path to the source file,
        and the value is the destination folder (path) where the file should be copied.

    Returns
    -------
    None
    """
    for source_file, dest_folder in destination_mapping.items():
        # Ensure destination folder exists
        os.makedirs(dest_folder, exist_ok=True)
        # Get the file name from the source file path
        file_name = os.path.basename(source_file)
        # Construct the destination file path
        dest_file = os.path.join(dest_folder, file_name)
        try:
            shutil.copy(source_file, dest_file)
            print(f"Copied '{source_file}' to '{dest_file}'")
        except Exception as e:
            print(f"Error copying '{source_file}' to '{dest_file}': {e}")


# Determine Number of Classes
train_dir = os.path.join(dataset_input, "train")
if not os.path.isdir(train_dir):
     raise FileNotFoundError(f"Training directory not found at {train_dir}")
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
num_classes = len(class_names)
if num_classes == 0:
     raise ValueError(f"No class subdirectories found in {train_dir}")
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

# Save class names
classes_txt_path = os.path.join(OUTPUT, f"classifier_classes_{model_name}.txt")
with open(classes_txt_path, "w") as f:
    for cls in class_names:
        f.write(cls + "\n")
print(f"Classes saved to {classes_txt_path}")



# Model Initialization & Freezing
print("\n--- Initializing Model ---")
model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
model = model.to(device)
model_config = model.default_cfg
print(f"Model: {model_name}")
image_size = model_config['input_size'][1]
print(f"Required image size: {image_size}")

# --- Freeze Backbone Layers ---
print("Freezing backbone layers...")
for param in model.parameters():
    param.requires_grad = False

# --- Unfreeze Classifier Head ---
classifier_attr_names = ['classifier', 'fc', 'head']
found_classifier = False
for attr_name in classifier_attr_names:
    if hasattr(model, attr_name):
        classifier_layer = getattr(model, attr_name)
        # Handle cases where the classifier might be a Sequential block
        if isinstance(classifier_layer, nn.Module):
             print(f"Unfreezing parameters in: model.{attr_name}")
             for param in classifier_layer.parameters():
                 param.requires_grad = True
             found_classifier = True
             break  # Stop after finding the first one
        else:
             print(f"Warning: Found attribute '{attr_name}' but it's not an nn.Module. Cannot unfreeze automatically.")

if not found_classifier:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("WARNING: Could not automatically find and unfreeze classifier head.")
    print(f"         Check model structure and manually unfreeze: {classifier_attr_names}")
    print("         Training ALL parameters instead (standard finetuning).")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # If head not found, unfreeze everything to revert to standard finetuning
    for param in model.parameters():
        param.requires_grad = True

# Data Transforms & Loaders
print("\n--- Setting up Data Transforms and Loaders ---")


class CenterBiasedRandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.1), ratio=(0.3, 1.3), center_bias=0.7, value=0.0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.center_bias = center_bias
        self.value = value

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img

        if not isinstance(img, torch.Tensor):
             # Assuming input might be PIL Image if not Tensor yet
             img = transforms.functional.to_tensor(img)

        c, h, w = img.shape
        area = h * w

        for _ in range(10):  # Attempt to find a valid box
            erase_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            h_erase = int(round((erase_area * aspect_ratio) ** 0.5))
            w_erase = int(round((erase_area / aspect_ratio) ** 0.5))

            if h_erase < h and w_erase < w:
                # Bias erasing toward the center
                center_h = h // 2
                center_w = w // 2
                # Use uniform distribution around center biased by center_bias
                max_offset_h = int((1.0 - self.center_bias) * (h - h_erase) / 2)
                max_offset_w = int((1.0 - self.center_bias) * (w - w_erase) / 2)

                top_center = center_h - h_erase // 2
                left_center = center_w - w_erase // 2

                top_offset = random.randint(-max_offset_h, max_offset_h) if max_offset_h > 0 else 0
                left_offset = random.randint(-max_offset_w, max_offset_w) if max_offset_w > 0 else 0

                top = max(0, min(h - h_erase, top_center + top_offset))
                left = max(0, min(w - w_erase, left_center + left_offset))

                if self.value == 'random':
                    erase_value = torch.rand(c, h_erase, w_erase, device=img.device)
                elif isinstance(self.value, (int, float)):
                    erase_value = torch.full((c, h_erase, w_erase), float(self.value), device=img.device)
                elif isinstance(self.value, (tuple, list)):
                    # Ensure value matches number of channels
                    if len(self.value) != c:
                         raise ValueError(f"Erase value tuple/list length ({len(self.value)}) must match image channels ({c})")
                    erase_value = torch.tensor(self.value, device=img.device).view(c, 1, 1).expand(c, h_erase, w_erase)
                else:
                    raise ValueError("Invalid value for erasing")

                img[:, top:top + h_erase, left:left + w_erase] = erase_value
                return img  # Erasing done

        return img  # fallback if no valid box found after 10 attempts


# Using Torchvision AutoAugment + RandomErasing
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(image_size), # Resize first if needed, AutoAugment expects consistent size
        transforms.RandomHorizontalFlip(), # Basic flip
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET), # AutoAugment policy
        transforms.ToTensor(), # Convert PIL to Tensor
        transforms.Normalize(model_config['mean'], model_config['std']), # Normalize after ToTensor
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0, inplace=False), # Standard Random Erasing
    ]),
    'val': transforms.Compose([
        # Validation transforms from timm recipe are often good
        transforms.Resize(int(image_size / model_config['crop_pct'] * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(model_config['mean'], model_config['std']),
    ]),
}


# --- Create Datasets and Dataloaders ---
try:
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(dataset_input, x), data_transforms[x])
        for x in ['train', 'val']
    }

    # Check if datasets are empty
    if not image_datasets['train'] or not image_datasets['val']:
         raise ValueError("Training or validation dataset is empty. Check data preparation.")

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=_batch_size, shuffle=(x == 'train'),
                      num_workers=_num_workers, pin_memory=True, drop_last=(x == 'train')) # drop_last for train
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(f"Dataset sizes: Train={dataset_sizes['train']}, Val={dataset_sizes['val']}")
except Exception as e:
     print(f"Error creating datasets/dataloaders: {e}")
     raise

# Helper Functions
def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    fig, ax = plt.subplots(figsize=(max(8, len(classes)//2), max(6, len(classes)//2.5))) # Adjust size based on num classes
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set(
        xticks=tick_marks,
        yticks=tick_marks,
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor") # Adjust rotation

    # Adjust text color threshold dynamically
    thresh = cm.max() * 0.6 # Adjust threshold for text color contrast
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8) # Smaller font size for dense matrices
    plt.tight_layout()
    return fig

# Training Loop (including unfreezing)

def train_model(model, criterion, initial_optimizer, initial_scheduler, num_epochs=25, early_stop_patience=3,
                unfreeze_epoch=-1, finetune_lr=1e-5, weight_decay=1e-5, mixup_fn=None):
    """
    Trains/validates model with AMP, optional backbone unfreezing, and optimizer/scheduler reset.
    """
    since = time.time()
    best_model_wts = model.state_dict().copy()
    best_acc = 0.0
    epochs_no_improve = 0

    # Use the passed initial optimizer and scheduler
    optimizer = initial_optimizer
    scheduler = initial_scheduler

    # 1. Initialize GradScaler
    # Creates a GradScaler once at the beginning of training.
    scaler = torch.amp.GradScaler(device='cuda')
    print(f"\n--- Starting Training (AMP Enabled: {scaler.is_enabled()}, Mixup/Cutmix Active: {mixup_fn is not None and (mixup_fn.mixup_alpha > 0. or mixup_fn.cutmix_alpha > 0.)}) ---")


    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # --- Check for Unfreezing ---
        if epoch == unfreeze_epoch:
            print(f"*** Unfreezing all layers at epoch {epoch + 1} ***")
            for param in model.parameters():
                param.requires_grad = True

            print(f"Re-initializing optimizer with LR: {finetune_lr} and Weight Decay: {weight_decay}")
            optimizer = optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=weight_decay)

            print(f"Re-initializing scheduler (CosineAnnealing) for remaining {num_epochs - epoch} epochs.")
            scheduler = CosineAnnealingLR(optimizer, T_max=(num_epochs - epoch), eta_min=finetune_lr/100)

            # Reset GradScaler's state upon optimizer change - Recommended when optimizer is re-initialized mid-training
            scaler = torch.amp.GradScaler(device='cuda')
            print("GradScaler state reset.")


        start_lr = optimizer.param_groups[0]['lr']
        print(f"Start of Epoch {epoch + 1}, Learning Rate: {start_lr:.7f}")

        train_acc, val_acc = None, None
        train_loss, val_loss = None, None

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            num_batches = len(dataloaders[phase])

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Apply Mixup/CutMix ONLY during training, BEFORE autocast
                labels_for_loss = labels # Default to original labels
                if phase == 'train' and mixup_fn is not None:
                    inputs, labels_for_loss = mixup_fn(inputs, labels) # labels_for_loss now contains soft/mixed targets

                # Zero gradients BEFORE the AMP context and forward pass
                optimizer.zero_grad()

                # Use autocast only for the training phase forward pass and loss calculation
                if phase == 'train':
                    # 2. Wrap forward pass + loss with autocast
                    with autocast(device_type='cuda'):
                        outputs = model(inputs)
                        # Use labels_for_loss (potentially mixed) for loss
                        loss = criterion(outputs, labels_for_loss)

                    # 3. Scale loss and call backward
                    scaler.scale(loss).backward()

                    # 4. Scaler step and update
                    scaler.step(optimizer)
                    scaler.update()

                    # Calculate predictions after potential optimizer step might have occurred
                    # (Usually calculated before step, but loss is primary driver for backward/step)
                    with torch.no_grad(): # Ensure preds calculation doesn't track gradients
                         _, preds = torch.max(outputs, 1)

                else:  # Validation phase - No GradScaler
                    with torch.no_grad():
                         outputs = model(inputs)
                         # Loss calculation for logging/monitoring
                         loss = criterion(outputs, labels)
                         _, preds = torch.max(outputs, 1)

                # Statistics Update
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # --- Logging and Best Model Check (Validation Phase) ---
            if phase == 'train':
                train_acc = epoch_acc
                train_loss = epoch_loss
                writer.add_scalar('Loss/train', epoch_loss, epoch)
            else: # phase == 'val'
                val_acc = epoch_acc
                val_loss = epoch_loss
                writer.add_scalar('Loss/val', epoch_loss, epoch)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict().copy()
                    epochs_no_improve = 0
                    print(f"*** Validation accuracy improved to {best_acc:.4f}, saving model weights. ***")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).")

                # Confusion Matrix Logic
                all_preds_cm = []
                all_labels_cm = []
                model.eval()
                with torch.no_grad():
                    for inputs_cm, labels_cm in dataloaders['val']:
                        inputs_cm = inputs_cm.to(device)
                        outputs_cm = model(inputs_cm)
                        _, preds_cm = torch.max(outputs_cm, 1)
                        all_preds_cm.extend(preds_cm.cpu().numpy())
                        all_labels_cm.extend(labels_cm.cpu().numpy())
                if len(all_labels_cm) > 0 and len(all_preds_cm) > 0:
                    try:
                        cm = confusion_matrix(all_labels_cm, all_preds_cm, labels=range(len(class_names)))
                        fig = plot_confusion_matrix(cm, classes=class_names, title=f'Epoch {epoch + 1} Confusion Matrix')
                        writer.add_figure(f'zz_Confusion_Matrix/Epoch_{epoch + 1}', fig, global_step=epoch)
                        latest_cm_path = os.path.join(OUTPUT, f"last_confusion_matrix_{model_name}.png")
                        fig.savefig(latest_cm_path)
                        plt.close(fig)
                    except Exception as e:
                        print(f"Error generating/saving confusion matrix: {e}")
                else:
                    print("Not enough data for confusion matrix.")
                # --- End Val Phase Logic ---

        # --- End of Epoch ---
        if train_acc is not None and val_acc is not None:
            writer.add_scalars('Accuracy', {'train': train_acc.item(), 'val': val_acc.item()}, epoch)
        if train_loss is not None and val_loss is not None:
            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)

        # Step the scheduler *once* per epoch
        if scheduler:
             scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Learning Rate", current_lr, epoch)
        writer.add_scalar("AMP/grad_scale", scaler.get_scale(), epoch)

        writer.flush()

        if epochs_no_improve >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break
    # --- End of Training Loop ---

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Accuracy: {best_acc:.4f}')

    print("Loading best model weights...")
    model.load_state_dict(best_model_wts)
    return model


# === SETUP BEFORE TRAINING ===
# --- Initialize loss ---
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# --- Initialize Mixup ---
print("\n--- Initializing Mixup/CutMix ---")
mixup_fn = Mixup(
    mixup_alpha=0.8,            # Mixup alpha, 0. to disable
    cutmix_alpha=1.0,           # Cutmix alpha, 0. to disable
    cutmix_minmax=None,         # Cutmix min/max ratio, None=defaults
    prob=1.0,                   # Probability of applying mixup or cutmix per batch
    switch_prob=0.5,            # Probability of switching to cutmix instead of mixup
    mode='batch',               # Perform mixup/cutmix on batch level
    label_smoothing=0.0,        # <<< Set to 0.0 as criterion handles it
    num_classes=num_classes
)
print(f"Mixup/CutMix active (Mixup alpha: {mixup_fn.mixup_alpha}, Cutmix alpha: {mixup_fn.cutmix_alpha})")


# --- Initial Optimizer (Only for Head) ---
# Filter parameters to train only the head initially
head_parameters = filter(lambda p: p.requires_grad, model.parameters())
num_head_params = sum(p.numel() for p in head_parameters if p.requires_grad)  # Recalculate after filter
if num_head_params == 0 and found_classifier:  # Check if head params were actually found and unfrozen
     print("WARNING: No trainable parameters found after attempting to unfreeze head. Reverting to full model training.")
     for param in model.parameters():
          param.requires_grad = True
     initial_optimizer = optim.AdamW(model.parameters(), lr=_learning_rate, weight_decay=_adamW_weight_decay)
else:
     print(f"Initializing optimizer for head parameters ({num_head_params} trainable) with LR: {_learning_rate}")
     initial_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=_learning_rate, weight_decay=_adamW_weight_decay)


# --- Initial Scheduler (Covers warm-up and potentially initial head training phase) ---
if _warmup_epochs >= _num_epochs:
     raise ValueError("_warmup_epochs must be less than _num_epochs")
if _learning_rate <= 0 or _initial_lr < 0:
     raise ValueError("Learning rates must be positive")
start_factor = _initial_lr / _learning_rate if _learning_rate > 0 else 0

# Scheduler for the entire duration, it will be replaced upon unfreezing
initial_warmup_scheduler = LinearLR(initial_optimizer, start_factor=start_factor, total_iters=_warmup_epochs)
# Cosine decay for the *full* duration after warmup, even though it will be replaced later
initial_main_scheduler = CosineAnnealingLR(initial_optimizer, T_max=(_num_epochs - _warmup_epochs), eta_min=1e-7)
initial_scheduler = SequentialLR(initial_optimizer, schedulers=[initial_warmup_scheduler, initial_main_scheduler], milestones=[_warmup_epochs])

# === CALL TRAINING ===
trained_model = train_model(
    model=model,
    criterion=criterion,
    initial_optimizer=initial_optimizer,        # Pass the head optimizer
    initial_scheduler=initial_scheduler,        # Pass the initial scheduler
    num_epochs=_num_epochs,
    early_stop_patience=_early_stop_patience,
    unfreeze_epoch=_unfreeze_epoch,             # Pass the epoch number to unfreeze
    finetune_lr=_finetune_lr,                   # Pass the LR for full finetuning
    weight_decay=_adamW_weight_decay,           # Pass weight decay for re-initialization
    mixup_fn=mixup_fn                           # Pass the mixup function

)


# 7. Save & Export
print("\n--- Saving and Exporting Model ---")
model_save_path = os.path.join(OUTPUT, f'{model_name}_model_timm.pth')
torch.save(trained_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

def export_finetuned_model(onnx_model_path, checkpoint_path, model_name, image_size, num_classes):  # Pass params explicitly
    device_cpu = torch.device("cpu")
    try:
        # Recreate model architecture on CPU
        model_ft = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        # Load the saved weights
        model_ft.load_state_dict(torch.load(checkpoint_path, map_location=device_cpu))
        model_ft.eval()  # Set to evaluation mode

        # Create dummy input matching expected size
        dummy_input = torch.randn(1, 3, image_size, image_size, device=device_cpu)

        # Export to ONNX
        torch.onnx.export(
            model_ft,
            dummy_input,
            onnx_model_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11,
            export_params=True  # Ensure weights are embedded
        )
        print(f"Fine-tuned model exported to ONNX format at: {onnx_model_path}")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")


onnx_model_path = os.path.join(OUTPUT, f"classifier_best_{model_name}.onnx")
export_finetuned_model(onnx_model_path, model_save_path, model_name, image_size, num_classes)

writer.close()

# 8. GitHub Repo Prep & Final Inference
print("\n--- Post-Training Operations ---")

# Save Run Settings
try:
    # Update settings extraction for scheduler if needed (might be CosineAnnealingLR at the end)
    final_optimizer_state = initial_optimizer.state_dict()  # Get state from the *last* optimizer used in train_model
    final_scheduler_state = initial_scheduler.state_dict() if initial_scheduler else None  # Get state from the *last* scheduler

    # Log parameters used during the run
    info_manager_current_run.add_key_value("model_name", model_name)
    info_manager_current_run.add_key_value("image_size", image_size)
    info_manager_current_run.add_key_value("num_classes", num_classes)
    info_manager_current_run.add_key_value("batch_size", _batch_size)
    info_manager_current_run.add_key_value("num_epochs_total", _num_epochs)
    info_manager_current_run.add_key_value("warmup_epochs", _warmup_epochs)
    info_manager_current_run.add_key_value("unfreeze_epoch", _unfreeze_epoch)
    info_manager_current_run.add_key_value("initial_head_lr", _learning_rate)
    info_manager_current_run.add_key_value("full_finetune_lr", _finetune_lr)
    info_manager_current_run.add_key_value("adamW_weight_decay", _adamW_weight_decay)
    info_manager_current_run.add_key_value("early_stop_patience", _early_stop_patience)
    info_manager_current_run.add_key_value("criterion", {"type": criterion.__class__.__name__, "label_smoothing": getattr(criterion, 'label_smoothing', 'N/A')})
    # Simple representation for optimizer/scheduler type used
    info_manager_current_run.add_key_value("optimizer_type", "AdamW")
    info_manager_current_run.add_key_value("scheduler_type", "SequentialLR (LinearWarmup + CosineAnnealing)")

    print("Run settings logged.")

except Exception as e:
    print(f"Error logging run settings: {e}")

