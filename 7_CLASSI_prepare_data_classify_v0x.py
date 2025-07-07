import os
import platform
import time
import shutil
from collections import Counter
import pandas as pd
import json
import cv2
import random
import numpy as np
import yaml
from utils.info_to_json import InfoJSON
from sklearn.model_selection import train_test_split

cpu_count = os.cpu_count() or 2
_max_workers = max(cpu_count - 1, 1)

file_path = "./training_settigns_info.json"  # Path for general settings
model_name = 'efficientnet_b2'

model_sizes = {
    "efficientnet_b0": 224,
    "efficientnet_b1": 240,
    "efficientnet_b2": 256,
    "efficientnet_b3": 288
}
classifier_image_size = model_sizes.get(model_name, 224)

# Create an instance of InfoJSON.
try:
    info_manager = InfoJSON(file_path)
    _min_samples_per_class_to_include_in_training = info_manager.get_key(
        "_min_samples_per_class_to_include_in_training", default=10)
    _max_samples_per_class = info_manager.get_key("_max_samples_per_class", default=500)
    # Load the list of species to use, default to empty list if not found or error
    _species_to_use_list = info_manager.get_key("species_to_use", default=[])

except FileNotFoundError:
    print(f"Warning: Settings file {file_path} not found. Using default values.")
    _min_samples_per_class_to_include_in_training = 10
    _max_samples_per_class = 500
    _species_to_use_list = []  # Default if file not found
except Exception as e:
    print(f"Error reading settings file {file_path}: {e}. Using default values.")
    _min_samples_per_class_to_include_in_training = 10
    _max_samples_per_class = 500
    _species_to_use_list = []  # Default on other errors

print(f"Min annotations per class threshold: {_min_samples_per_class_to_include_in_training}")
print(f"Max images (crops) per class target: {_max_samples_per_class}")
if _species_to_use_list:
    print(f"Explicit species list provided: {_species_to_use_list}")
else:
    print("No explicit species list provided (using threshold only).")

from concurrent.futures import ProcessPoolExecutor, as_completed
import imgaug.augmenters as iaa

# Patch for imgaug compatibility with NumPy >= 1.20
try:
    np.bool = bool
except AttributeError:
    pass  # Avoid error if np.bool doesn't exist or isn't bool

# >>> Configuration >>>
_subset_for_testing = False  # Whether to use a subset for testing
_subset_size = 100  # Size of the subset if _subset_for_testing is True
_test_ratio = 0.025  # Proportion for the test set
_val_ratio = 0.2  # Proportion for the validation set
# Train ratio is calculated based on test and val
_train_ratio = round(1.0 - _test_ratio - _val_ratio, 2)

if _train_ratio <= 0:
    raise ValueError("Test and Validation ratios sum to 1.0 or more. Decrease them.")
print(
    f"Target split ratios: train={_train_ratio}, val={_val_ratio}, test={_test_ratio} (sum={_train_ratio + _val_ratio + _test_ratio})")

_balance_only_training_set = True  # Only augment training set
_cap_val_test_sets = True  # Apply downsampling to val/test if they exceed _max_samples_per_class

# >>> Paths >>>
if platform.system() == "Darwin":
    DATA_BASE_PATH = "..."
elif platform.system() == "Linux":
    DATA_BASE_PATH = "..."
else:
    raise OSError("Unsupported operating system detected.")

DATA_BASE_DIR = DATA_BASE_PATH
IMAGE_UNPROCESSED_DIR = os.path.join(DATA_BASE_DIR, "images", "unprocessed")
METADATA_FILE = os.path.join(DATA_BASE_DIR, "metadata", "dataset.csv")
TRAIN_FOLDER = os.path.join(DATA_BASE_PATH, "training")

# --- Refactored Output Folders ---
CLASS_BASE_FOLDER = os.path.join(TRAIN_FOLDER, "classification")
TEMP_FOLDER = os.path.join(CLASS_BASE_FOLDER, "temp")
FINAL_FOLDER = os.path.join(CLASS_BASE_FOLDER, "images")  # Final dataset location

os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(FINAL_FOLDER, exist_ok=True)

file_path_current_run = os.path.join(CLASS_BASE_FOLDER, "training_settings_info_current_run.json")


# >>> Helper Functions >>>
def create_dataset_reference_json(images_dir, reference_filename="dataset_reference.json"):
    """
    Creates dataset reference JSON file.

    Args:
        images_dir (str): Path to the images directory containing splits.
        reference_filename (str): Name of the output JSON file (default: "dataset_reference.json").

    Returns:
        None
    """
    dataset_reference = {}
    for split in ["train", "val", "test"]:
        split_path = os.path.join(images_dir, split)
        split_dict = {}
        if os.path.exists(split_path):
            species_folders = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            for species in sorted(species_folders):
                species_path = os.path.join(split_path, species)
                images_list = [f for f in sorted(os.listdir(species_path))
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                split_dict[species] = images_list
        dataset_reference[split] = split_dict

    reference_file_path = os.path.join(images_dir, reference_filename)
    with open(reference_file_path, "w") as f:
        json.dump(dataset_reference, f, indent=4)
    print(f"Reference JSON file created at {reference_file_path}")


def show_dataset_stats(reference_file_path):
    """
    Displays dataset statistics.

    Args:
        reference_file_path (str): Path to the dataset reference JSON file.

    Returns:
        None
    """
    try:
        with open(reference_file_path, 'r') as f:
            dataset_reference = json.load(f)
    except FileNotFoundError:
        print(f"Error: Statistics file not found at {reference_file_path}")
        return

    total_images = 0
    overall_species = {}
    print("\nDataset Statistics:")

    for split, species_mapping in dataset_reference.items():
        split_total = 0
        print(f"\nSplit: {split}")
        if not species_mapping:
            print("  No species found in this split.")
            continue
        for species, images in species_mapping.items():
            count = len(images)
            print(f"  {species}: {count} images")
            split_total += count
            overall_species[species] = overall_species.get(species, 0) + count
        total_images += split_total
        print(f"Total images in {split}: {split_total}")

    print("\nOverall Species Counts (across all splits):")
    if not overall_species:
        print("  No species found overall.")
    else:
        for species, count in sorted(overall_species.items()):
            print(f"  {species}: {count} images")

    print(f"\nTotal images in dataset: {total_images}")


def create_square_crop(image, bbox, margin_percent=0.2, pad_color=(0, 0, 0)):
    """
    Creates a square crop from an image based on bounding box.

    Args:
        image (numpy.ndarray): Input image array.
        bbox (tuple): Bounding box coordinates as (x_min, y_min, x_max, y_max).
        margin_percent (float): Fractional margin to add around the bounding box.
        pad_color (tuple): RGB color for padding.

    Returns:
        numpy.ndarray: Square cropped image.
    """
    bx1, by1, bx2, by2 = bbox
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    bbox_width = bx2 - bx1
    bbox_height = by2 - by1
    bbox_side = max(bbox_width, bbox_height)
    new_side = int(bbox_side * (1 + margin_percent))
    desired_x1 = int(cx - new_side / 2)
    desired_y1 = int(cy - new_side / 2)
    desired_x2 = desired_x1 + new_side
    desired_y2 = desired_y1 + new_side
    image_h, image_w = image.shape[:2]
    crop_x1 = max(0, desired_x1)
    crop_y1 = max(0, desired_y1)
    crop_x2 = min(image_w, desired_x2)
    crop_y2 = min(image_h, desired_y2)
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    pad_left = crop_x1 - desired_x1
    pad_top = crop_y1 - desired_y1
    pad_right = desired_x2 - crop_x2
    pad_bottom = desired_y2 - crop_y2

    # Ensure padding values are non-negative
    pad_left = max(0, pad_left)
    pad_top = max(0, pad_top)
    pad_right = max(0, pad_right)
    pad_bottom = max(0, pad_bottom)

    square_crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                     borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return square_crop


# Worker function for creating crops for a specific split
def process_row_for_split_crops(row_tuple):
    """
    Reads an image row, extracts crops for valid annotations, and saves
    them to the specified temporary split/species directory.

    Args:
        row_tuple: A tuple containing (index, row_data, temp_split_dir, valid_species_set).
                   row_data is a pandas Series.

    Returns:
        List of tuples: [(crop_path, species_name), ...] for crops created from this row.
    """
    index, row, temp_split_dir, valid_species_set = row_tuple
    processed_crops = []

    if not row["approved_annotation"]:
        return processed_crops

    try:
        annotation_json = json.loads(row["approved_annotation"])
    except Exception as e:
        return processed_crops  # Soft fail for parsing errors

    if "images" not in annotation_json or not annotation_json["images"]:
        return processed_crops
    image_entry = annotation_json["images"][0]

    image_filename = os.path.join(DATA_BASE_DIR, row["image_path"])
    if not os.path.isfile(image_filename):
        return processed_crops

    image_cv2 = cv2.imread(image_filename)
    if image_cv2 is None:
        return processed_crops
    h, w = image_cv2.shape[:2]
    if w == 0 or h == 0:
        return processed_crops

    # Get image dimensions if missing in JSON
    img_width = image_entry.get("width", w)
    img_height = image_entry.get("height", h)
    if img_width == 0 or img_height == 0:
        img_width, img_height = w, h

    # --- Extract crops ---
    base_name = os.path.splitext(os.path.basename(image_filename))[0]
    annotations = annotation_json.get("annotations", [])
    categories = annotation_json.get("categories", [])
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    for idx, ann in enumerate(annotations):
        category_id = ann.get("category_id")
        species_name = cat_id_to_name.get(category_id)

        # Skip if species is not valid or category ID is missing
        if species_name is None or species_name not in valid_species_set:
            continue

        # Assume ann["bbox"] is [x_min, y_min, width, height] (COCO format)
        try:
            x_min, y_min, bbox_w, bbox_h = map(float, ann["bbox"])
            # Basic validation for bbox coordinates
            if bbox_w <= 0 or bbox_h <= 0 or x_min < 0 or y_min < 0 or x_min + bbox_w > img_width or y_min + bbox_h > img_height:
                continue
            bbox_pascal = (x_min, y_min, x_min + bbox_w, y_min + bbox_h)
        except (ValueError, TypeError, KeyError) as e:
            continue

        try:
            # Create and resize crop
            cropped = create_square_crop(image_cv2, bbox_pascal, margin_percent=0.1, pad_color=(0, 0, 0))
            if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                continue
            resized_cropped = cv2.resize(cropped, (classifier_image_size, classifier_image_size),
                                         interpolation=cv2.INTER_AREA)

            # Save crop to the correct temp split/species folder
            species_folder = os.path.join(temp_split_dir, species_name)
            os.makedirs(species_folder, exist_ok=True)
            crop_filename = f"{base_name}_ann{idx}.jpg"
            crop_path = os.path.join(species_folder, crop_filename)
            cv2.imwrite(crop_path, resized_cropped)
            processed_crops.append((crop_path, species_name))
        except Exception as e:
            # Catch errors during cropping, resizing, or saving
            print(f"❌ Row {index}, Ann {idx}: Error processing crop for {species_name}: {e}")

    return processed_crops


# >>> Augmentation Functions >>>
def process_single_image_augmentation(image_tuple):
    """
    Processes augmentation for one image crop file.

    Args:
        image_tuple (tuple): A tuple containing (filename, species_folder, seq, num_augmentations).

    Returns:
        bool: True if augmentation succeeded, False otherwise.
    """
    filename, species_folder, seq, num_augmentations = image_tuple
    image_path = os.path.join(species_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Augmentation - Could not read image {filename} in {species_folder}. Skipping.")
        return False
    try:
        base, ext = os.path.splitext(filename)
        for i in range(num_augmentations):
            aug_image = seq(image=image)  # Apply the augmentation sequence
            # Ensure unique names even if multiple augmentations happen over time
            timestamp = int(time.time() * 1000)  # Add timestamp for uniqueness
            new_filename = f"{base}_aug{i}_{timestamp}{ext}"
            output_image_path = os.path.join(species_folder, new_filename)
            cv2.imwrite(output_image_path, aug_image)
        return True
    except Exception as e:
        print(f"Error during augmentation for {filename} in {species_folder}: {e}")
        return False


def get_augmentation_sequence():
    """
    Defines and returns the imgaug augmentation sequence.

    Returns:
        imgaug.Sequential: The augmentation sequence to apply.
    """
    seq = iaa.Sequential([
        iaa.Affine(scale=(0.9, 1.1),
                   rotate=(-15, 15),
                   translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                   shear=(-5, 5)),
        iaa.Fliplr(0.5),
        iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-20, 20)),  # More robust brightness/contrast
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.75))),  # Slightly increased blur possibility
        iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255))),
        iaa.LinearContrast((0.8, 1.2)),  # Wider contrast range
    ], random_order=True)  # Randomize order of augmenters
    return seq


# >>> Balancing Functions >>>
def downsample_species_folder(species_folder, target_count):
    """
    Deletes excess image files in a species folder to reach the target count.

    Args:
        species_folder (str): Path to the species image folder.
        target_count (int): Desired maximum number of images.

    Returns:
        None
    """
    files = [f for f in os.listdir(species_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(files)
    if current_count > target_count:
        to_remove_count = current_count - target_count
        files_to_remove = random.sample(files, to_remove_count)
        removed_count = 0
        for f in files_to_remove:
            try:
                os.remove(os.path.join(species_folder, f))
                removed_count += 1
            except OSError as e:
                print(f"Error deleting file {f} during downsampling: {e}")
        print(f"Downsampled '{os.path.basename(species_folder)}': Removed {removed_count} files.")
    # else: No action needed if count is already <= target_count


def upsample_species_folder(species_folder, target_count, augmentation_sequence):
    """
    Generates augmented images to reach the target count in a species folder.

    Args:
        species_folder (str): Path to the species image folder.
        target_count (int): Desired number of images.
        augmentation_sequence (imgaug.Sequential): Sequence used for augmentations.

    Returns:
        None
    """
    files = [f for f in os.listdir(species_folder) if
             f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_aug' not in f]  # Base files only
    current_count = len(
        [f for f in os.listdir(species_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])  # Count all files

    if not files:  # No base images to augment from
        print(f"Warning: Cannot upsample '{os.path.basename(species_folder)}' - no base images found.")
        return

    if current_count < target_count:
        deficit = target_count - current_count

        # --- Augmentation Task Preparation ---
        tasks = []
        # Distribute augmentations: give each base image roughly equal number of augmentations
        num_base_files = len(files)
        augmentations_per_base = deficit // num_base_files
        remainder = deficit % num_base_files

        print(f"Upsampling '{os.path.basename(species_folder)}': Need {deficit} more images. Preparing tasks...")

        for i, base_filename in enumerate(files):
            num_aug_for_this_file = augmentations_per_base + (1 if i < remainder else 0)
            if num_aug_for_this_file > 0:
                tasks.append((base_filename, species_folder, augmentation_sequence, num_aug_for_this_file))

        if not tasks:
            print(
                f"Warning: No augmentation tasks created for {os.path.basename(species_folder)}, deficit was {deficit}")
            return

        # --- Execute Augmentation in Parallel ---
        successful_augmentations = 0
        with ProcessPoolExecutor(max_workers=_max_workers) as executor:
            results = executor.map(process_single_image_augmentation, tasks)
            successful_augmentations = sum(1 for result in results if result)

        print(
            f"Upsampling '{os.path.basename(species_folder)}': Generated {successful_augmentations} augmented images.")

        # Final check and potential minor downsample if overshoot occurred (e.g., due to rounding)
        final_files = [f for f in os.listdir(species_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(final_files) > target_count:
            print(f"Minor overshoot in {os.path.basename(species_folder)}, downsampling...")
            downsample_species_folder(species_folder, target_count)


def balance_split_directory(split_dir, max_samples, augment=False):
    """
    Balances a split directory by downsampling or upsampling species folders.

    Args:
        split_dir (str): Path to the split directory (train, val, or test).
        max_samples (int): Maximum samples per species.
        augment (bool): Whether to apply augmentation when undersampled.

    Returns:
        None
    """
    print(f"\nBalancing directory: {split_dir} (Max Samples: {max_samples}, Augment: {augment})")
    if not os.path.isdir(split_dir):
        print("Directory not found.")
        return

    augmentation_seq = get_augmentation_sequence() if augment else None

    species_folders = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]

    if not species_folders:
        print("No species subdirectories found.")
        return

    for species in species_folders:
        species_path = os.path.join(split_dir, species)
        current_files = [f for f in os.listdir(species_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(current_files)
        print(f"  Species '{species}': Found {current_count} images.")

        if current_count > max_samples:
            downsample_species_folder(species_path, max_samples)
        elif current_count < max_samples and augment:
            upsample_species_folder(species_path, max_samples, augmentation_seq)
        # else: count is <= max_samples and augmentation is off, or count == max_samples - do nothing


# >>> Main Pipeline Logic >>>
def main_pipeline():
    """
    Executes the refactored pipeline: split metadata, create crops,
    balance splits, and finalize dataset.

    Returns:
        None
    """

    start_time = time.time()

    # --- Step 0: Preparation ---
    # Clean potential leftover temp and final directories
    if os.path.exists(TEMP_FOLDER):
        print(f"Removing existing temporary folder: {TEMP_FOLDER}")
        shutil.rmtree(TEMP_FOLDER)
    if os.path.exists(FINAL_FOLDER):
        print(f"Removing existing final folder: {FINAL_FOLDER}")
        shutil.rmtree(FINAL_FOLDER)
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    os.makedirs(FINAL_FOLDER, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(TEMP_FOLDER, split), exist_ok=True)
        os.makedirs(os.path.join(FINAL_FOLDER, split), exist_ok=True)

    # --- Step 1: Load Metadata and Determine Valid Species ---
    print("\n--- Step 1: Loading Metadata and Determining Valid Species ---")
    try:
        df = pd.read_csv(METADATA_FILE, keep_default_na=False)
    except FileNotFoundError:
        print(f"FATAL: Metadata file not found at {METADATA_FILE}")
        return

    # --- Basic filtering ---
    if "image_hash" in df.columns:
        df.drop_duplicates(subset=["image_hash"], inplace=True)
    df = df[df["approved_annotation_downloaded"] == "yes"].reset_index(drop=True)
    df = df[df["approved_annotation"].notna() & (df["approved_annotation"] != '')] # Ensure annotation JSON string exists

    if df.empty:
        print("❌ FATAL: No rows with downloaded/non-empty annotations.")
        return

    if _subset_for_testing:
        print(f"🔍 Using subset of size: {_subset_size}")
        df = df.sample(n=min(len(df), _subset_size), random_state=42).reset_index(drop=True)

    # --- Count species annotations ---
    print("Counting species from annotations...")
    species_annotation_counts = Counter()
    rows_with_species_info = {}  # Store {index: (set_of_species, first_species)}

    for index, row in df.iterrows():
        species_in_row = set()
        first_species_for_row = None
        annotation_str = row.get("approved_annotation")
        if not annotation_str: continue

        try:
            ann_json = json.loads(annotation_str)
            categories = {cat["id"]: cat["name"] for cat in ann_json.get("categories", [])}
            annotations = ann_json.get("annotations", [])
            if not categories: continue  # Skip if no categories defined

            for ann in annotations:
                cat_id = ann.get("category_id")
                species_name = categories.get(cat_id)
                if species_name:
                    species_in_row.add(species_name)
                    if first_species_for_row is None:
                         first_species_for_row = species_name  # Store first for potential stratification key
            # Update counts after processing all annotations for the row
            species_annotation_counts.update(species_in_row)
            if species_in_row:
                 rows_with_species_info[index] = (species_in_row, first_species_for_row)

        except (json.JSONDecodeError, TypeError, KeyError):
             pass  # Ignore parsing errors during counting

    print(f"Found {len(species_annotation_counts)} unique species across all annotations.")
    if not species_annotation_counts: print("❌ FATAL: No species found in any annotations."); return


    # Determine FINAL valid species using threshold AND optional list
    # 1. Apply the annotation count threshold
    valid_species_based_on_threshold = {
        sp for sp, count in species_annotation_counts.items() if count >= _min_samples_per_class_to_include_in_training
    }
    print(f"Found {len(valid_species_based_on_threshold)} species meeting threshold >= {_min_samples_per_class_to_include_in_training}: {sorted(list(valid_species_based_on_threshold))}")

    # 2. Apply the species filter list if it's provided and not empty
    if _species_to_use_list: # Check if the list loaded from settings is not empty
        species_to_use_set = set(_species_to_use_list)
        print(f"Applying filter based on provided list: {sorted(list(species_to_use_set))}")

        # Find the intersection
        valid_species = valid_species_based_on_threshold.intersection(species_to_use_set)

        # Report species that were in the list but didn't meet the threshold
        ignored_from_list = species_to_use_set - valid_species_based_on_threshold
        if ignored_from_list:
            print(f"⚠️ Warning: The following species from the provided list did NOT meet the annotation threshold ({_min_samples_per_class_to_include_in_training}) and were excluded: {sorted(list(ignored_from_list))}")

        # Report species that met threshold but were not in the list
        excluded_by_list = valid_species_based_on_threshold - species_to_use_set
        if excluded_by_list:
             print(f"ℹ️ Info: The following species met the threshold but were excluded because they were NOT in the provided list: {sorted(list(excluded_by_list))}")

    else:
        # If no list provided, use only the threshold result
        print("No species filter list provided. Using all species that meet the annotation threshold.")
        valid_species = valid_species_based_on_threshold


    if not valid_species:
        print(f"❌ FATAL: No species remaining after applying threshold and/or filter list.")
        return
    print(f"==> Final set of {len(valid_species)} valid species for dataset creation: {sorted(list(valid_species))}")
    valid_species_set = set(valid_species)  # Create the final set for lookups


    # --- Filter DataFrame to rows containing at least one *final* valid species ---
    valid_indices = [
        index for index, (species_set, _) in rows_with_species_info.items()
        if any(sp in valid_species_set for sp in species_set) # Check if ANY species in the row is in the final valid set
    ]
    df_filtered = df.loc[valid_indices].copy()

    # Add the stratification key column (using the first species found in the row, *if* that species is in the final valid_species_set)
    def get_stratify_key(row_index):
        if row_index in rows_with_species_info:
            _, first_species = rows_with_species_info[row_index]
            # Ensure the stratification key itself is a valid species
            return first_species if first_species in valid_species_set else None
        return None

    df_filtered['stratify_key'] = df_filtered.index.map(get_stratify_key)
    # Remove rows where stratification key couldn't be determined (should be rare if valid_indices logic is correct)
    df_filtered.dropna(subset=['stratify_key'], inplace=True)

    if df_filtered.empty:
        print("❌ FATAL: DataFrame is empty after filtering for rows containing final valid species.")
        return

    print(f"Filtered DataFrame to {len(df_filtered)} rows containing final valid species.")


    # --- Step 2: Split Metadata DataFrame ---
    print("\n--- Step 2: Splitting Metadata ---")
    stratify_column = df_filtered['stratify_key']

    # Check if stratification is possible
    unique_classes = stratify_column.nunique()
    class_counts = stratify_column.value_counts()
    min_class_count = class_counts.min() if not class_counts.empty else 0
    required_for_split = 2  # Need at least 2 samples per class for train/test split, more for train/val/test

    if unique_classes == 0 or min_class_count < required_for_split:
        print(
            f"Warning: Cannot stratify split with min class count {min_class_count} (required {required_for_split}). Performing non-stratified split.")
        stratify_data = None
    else:
        stratify_data = stratify_column

    # Split into Train and Temp (Val + Test)
    try:
        df_train, df_temp = train_test_split(
            df_filtered,
            test_size=(_val_ratio + _test_ratio),
            random_state=42,
            stratify=stratify_data
        )
    except ValueError as e:
        print(f"Warning: Stratified split failed ({e}). Falling back to non-stratified split.")
        df_train, df_temp = train_test_split(
            df_filtered,
            test_size=(_val_ratio + _test_ratio),
            random_state=42,
            stratify=None
        )

    # Split Temp into Val and Test
    # Adjust ratio for splitting the temp set
    relative_test_ratio = _test_ratio / (_val_ratio + _test_ratio) if (_val_ratio + _test_ratio) > 0 else 0

    # Check stratification possibility for the temp split
    stratify_data_temp = df_temp['stratify_key'] if stratify_data is not None else None
    if stratify_data_temp is not None:
        temp_class_counts = stratify_data_temp.value_counts()
        min_temp_class_count = temp_class_counts.min() if not temp_class_counts.empty else 0
        if df_temp.empty or stratify_data_temp.nunique() == 0 or min_temp_class_count < 2:  # Need at least 2 for val/test split if stratifying
            print("Warning: Cannot stratify val/test split. Performing non-stratified split.")
            stratify_data_temp = None

    if len(df_temp) > 0 and relative_test_ratio > 0 and relative_test_ratio < 1:
        try:
            df_val, df_test = train_test_split(
                df_temp,
                test_size=relative_test_ratio,
                random_state=42,
                stratify=stratify_data_temp
            )
        except ValueError as e:
            print(f"Warning: Stratified val/test split failed ({e}). Falling back to non-stratified split.")
            df_val, df_test = train_test_split(
                df_temp,
                test_size=relative_test_ratio,
                random_state=42,
                stratify=None
            )

    elif len(df_temp) > 0 and relative_test_ratio == 0:  # Only validation needed
        df_val = df_temp
        df_test = pd.DataFrame(columns=df_temp.columns)  # Empty test set
    elif len(df_temp) > 0 and relative_test_ratio == 1:  # Only test needed
        df_test = df_temp
        df_val = pd.DataFrame(columns=df_temp.columns)  # Empty val set
    else:  # df_temp was empty
        df_val = pd.DataFrame(columns=df_filtered.columns)
        df_test = pd.DataFrame(columns=df_filtered.columns)

    print(f"Metadata split: {len(df_train)} train, {len(df_val)} val, {len(df_test)} test rows.")

    # --- Step 3: Create Crops in Temporary Split Folders ---
    print("\n--- Step 3: Creating Crops in Temp Split Folders ---")
    valid_species_set = set(valid_species)  # Use set for faster lookups
    split_dfs = {'train': df_train, 'val': df_val, 'test': df_test}
    total_crops_created = 0

    for split_name, split_df in split_dfs.items():
        print(f"Processing {split_name} split...")
        temp_split_dir = os.path.join(TEMP_FOLDER, split_name)
        os.makedirs(temp_split_dir, exist_ok=True)

        # Prepare arguments for parallel processing
        tasks = [(index, row, temp_split_dir, valid_species_set) for index, row in split_df.iterrows()]

        if not tasks:
            print(f"No rows to process for {split_name} split.")
            continue

        split_crops_count = 0
        with ProcessPoolExecutor(max_workers=_max_workers) as executor:
            # Use map for simpler processing, results are lists of (crop_path, species_name)
            results = executor.map(process_row_for_split_crops, tasks)
            for crop_list in results:
                split_crops_count += len(crop_list)

        print(f"Created {split_crops_count} crops for {split_name} split.")
        total_crops_created += split_crops_count

    print(f"\nTotal initial crops created across all splits: {total_crops_created}")

    # --- Step 4: Balance Splits (Primarily Training Set) ---
    print("\n--- Step 4: Balancing Splits ---")

    # Balance Training Set (Augmentation enabled)
    balance_split_directory(
        os.path.join(TEMP_FOLDER, 'train'),
        max_samples=_max_samples_per_class,
        augment=True  # Always augment training set if needed
    )

    # Balance Validation Set (Optional Augmentation, Optional Capping)
    balance_split_directory(
        os.path.join(TEMP_FOLDER, 'val'),
        max_samples=_max_samples_per_class,
        augment=(not _balance_only_training_set)  # Augment only if specifically enabled
    )
    if _cap_val_test_sets and _balance_only_training_set:  # Apply capping if needed, even if not augmenting
        print(f"Applying capping (if needed) to val set...")
        balance_split_directory(os.path.join(TEMP_FOLDER, 'val'), max_samples=_max_samples_per_class, augment=False)

    # Balance Test Set (Optional Augmentation, Optional Capping)
    balance_split_directory(
        os.path.join(TEMP_FOLDER, 'test'),
        max_samples=_max_samples_per_class,
        augment=(not _balance_only_training_set)  # Augment only if specifically enabled
    )
    if _cap_val_test_sets and _balance_only_training_set:  # Apply capping if needed, even if not augmenting
        print(f"Applying capping (if needed) to test set...")
        balance_split_directory(os.path.join(TEMP_FOLDER, 'test'), max_samples=_max_samples_per_class, augment=False)

    # --- Step 5: Move Balanced Data to Final Location ---
    print("\n--- Step 5: Moving Balanced Data to Final Location ---")
    for split in ['train', 'val', 'test']:
        src_dir = os.path.join(TEMP_FOLDER, split)
        dest_dir = os.path.join(FINAL_FOLDER, split)
        if os.path.exists(src_dir):
            # Ensure destination exists (it should, but double-check)
            os.makedirs(dest_dir, exist_ok=True)
            print(f"Moving {src_dir} to {dest_dir}...")
            # Move species folders individually to handle existing destination
            for item in os.listdir(src_dir):
                s = os.path.join(src_dir, item)
                d = os.path.join(dest_dir, item)
                if os.path.isdir(s):
                    shutil.move(s, d)  # Move the whole species folder
            print(f"Move complete for {split} split.")
        else:
            print(f"Source directory {src_dir} not found, skipping move.")

    # --- Step 6: Cleanup and Final Stats ---
    print("\n--- Step 6: Cleanup and Final Report ---")
    try:
        print(f"Removing temporary folder: {TEMP_FOLDER}")
        shutil.rmtree(TEMP_FOLDER)
    except OSError as e:
        print(f"Error removing temporary folder {TEMP_FOLDER}: {e}")

    # Create reference JSON and show final stats
    create_dataset_reference_json(FINAL_FOLDER)
    reference_file_path = os.path.join(FINAL_FOLDER, "dataset_reference.json")
    show_dataset_stats(reference_file_path)

    # Save run settings
    print(f"Saving run settings to {file_path_current_run}")
    try:
        run_info = {
            "_min_samples_per_class_to_include_in_training": _min_samples_per_class_to_include_in_training,
            "_max_samples_per_class": _max_samples_per_class,
            "_train_ratio_requested": _train_ratio,
            "_val_ratio_requested": _val_ratio,
            "_test_ratio_requested": _test_ratio,
            "classifier_image_size": classifier_image_size,
            "model_name": model_name,
            "data_base_path": DATA_BASE_PATH,
            "final_dataset_path": FINAL_FOLDER,
            "balance_only_training_set": _balance_only_training_set,
            "cap_val_test_sets": _cap_val_test_sets,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        # Use InfoJSON utility if available, otherwise basic JSON dump
        try:
            info_manager_current_run = InfoJSON(file_path_current_run)
            for key, value in run_info.items():
                info_manager_current_run.add_key_value(key, value)
            print("Run settings saved using InfoJSON.")
        except NameError:
            with open(file_path_current_run, 'w') as f:
                json.dump(run_info, f, indent=4)
            print("Run settings saved using standard JSON.")

    except Exception as e:
        print(f"Error saving run settings: {e}")

    end_time = time.time()
    print(f"\nPipeline finished in {end_time - start_time:.2f} seconds.")


# =============================================================================
# Run Main Pipeline
# =============================================================================
if __name__ == "__main__":
    main_pipeline()
