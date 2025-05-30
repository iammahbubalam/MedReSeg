import pandas as pd
import os
import shutil

def create_subset(original_csv_path, original_data_root_dir, subset_base_dir, num_rows=100):
    """
    Creates a subset of the dataset.

    Args:
        original_csv_path (str): Path to the original CSV file.
        original_data_root_dir (str): Path to the root directory of the original dataset (containing images/masks folders).
        subset_base_dir (str): Path to the base directory where the subset will be created (e.g., project_root/dataset_subset/SAMed2Dv1).
        num_rows (int): The number of rows to attempt to include in the subset.
    """
    
    subset_csv_name = "subset_" + os.path.basename(original_csv_path)
    subset_csv_path = os.path.join(subset_base_dir, subset_csv_name)
    subset_images_dir = os.path.join(subset_base_dir, 'images')
    subset_masks_dir = os.path.join(subset_base_dir, 'masks')

    # Create output directories
    os.makedirs(subset_images_dir, exist_ok=True)
    os.makedirs(subset_masks_dir, exist_ok=True)
    print(f"Subset directories created/ensured at: {subset_base_dir}")

    # Read original CSV
    try:
        df = pd.read_csv(original_csv_path)
    except FileNotFoundError:
        print(f"Error: Original CSV not found at {original_csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Original CSV at {original_csv_path} is empty or invalid.")
        return

    if df.empty:
        print(f"Error: Original CSV at {original_csv_path} is empty.")
        return

    print(f"Successfully read original CSV: {original_csv_path} with {len(df)} rows.")

    # Select subset of rows
    df_subset_initial = df.head(num_rows)
    
    actual_rows_to_process = len(df_subset_initial)
    if actual_rows_to_process < num_rows:
        print(f"Warning: Original CSV has only {actual_rows_to_process} rows, requested {num_rows}. Processing {actual_rows_to_process} rows.")
    
    print(f"Attempting to process the first {actual_rows_to_process} rows for the subset.")

    valid_rows_data = []
    copied_image_count = 0
    copied_mask_count = 0
    skipped_count = 0

    for index, row in df_subset_initial.iterrows():
        img_rel_path = row.get('image_path')
        mask_rel_path = row.get('mask_path')

        if not img_rel_path or not mask_rel_path:
            print(f"Skipping row index {index} due to missing 'image_path' or 'mask_path'.")
            skipped_count += 1
            continue

        # Source paths
        src_img_full_path = os.path.join(original_data_root_dir, img_rel_path)
        src_mask_full_path = os.path.join(original_data_root_dir, mask_rel_path)

        # Destination paths (relative paths in CSV remain the same, base directory changes)
        dest_img_full_path = os.path.join(subset_base_dir, img_rel_path)
        dest_mask_full_path = os.path.join(subset_base_dir, mask_rel_path)

        valid_image_copied = False
        valid_mask_copied = False

        # Check and copy image
        if os.path.exists(src_img_full_path):
            if os.path.getsize(src_img_full_path) > 0:
                try:
                    # Ensure destination subdirectory exists (e.g., dataset_subset/SAMed2Dv1/images/)
                    os.makedirs(os.path.dirname(dest_img_full_path), exist_ok=True)
                    shutil.copy2(src_img_full_path, dest_img_full_path)
                    valid_image_copied = True
                    copied_image_count += 1
                except Exception as e:
                    print(f"Error copying image {src_img_full_path} to {dest_img_full_path}: {e}")
            else:
                print(f"Skipping 0KB image: {src_img_full_path} (for row index {index})")
        else:
            print(f"Skipping non-existent image: {src_img_full_path} (for row index {index})")

        # Check and copy mask
        if os.path.exists(src_mask_full_path):
            if os.path.getsize(src_mask_full_path) > 0:
                try:
                    # Ensure destination subdirectory exists (e.g., dataset_subset/SAMed2Dv1/masks/)
                    os.makedirs(os.path.dirname(dest_mask_full_path), exist_ok=True)
                    shutil.copy2(src_mask_full_path, dest_mask_full_path)
                    valid_mask_copied = True
                    copied_mask_count += 1
                except Exception as e:
                    print(f"Error copying mask {src_mask_full_path} to {dest_mask_full_path}: {e}")
            else:
                print(f"Skipping 0KB mask: {src_mask_full_path} (for row index {index})")
        else:
            print(f"Skipping non-existent mask: {src_mask_full_path} (for row index {index})")

        if valid_image_copied and valid_mask_copied:
            valid_rows_data.append(row.to_dict())
        else:
            skipped_count += 1
            print(f"Sample at original index {index} (image: {img_rel_path}, mask: {mask_rel_path}) was not fully copied.")
            # Clean up partially copied files for this sample
            if valid_image_copied and not valid_mask_copied and os.path.exists(dest_img_full_path):
                try:
                    os.remove(dest_img_full_path)
                    print(f"Cleaned up partially copied image: {dest_img_full_path}")
                    copied_image_count -=1
                except OSError as e:
                    print(f"Error cleaning up image {dest_img_full_path}: {e}")
            if valid_mask_copied and not valid_image_copied and os.path.exists(dest_mask_full_path):
                try:
                    os.remove(dest_mask_full_path)
                    print(f"Cleaned up partially copied mask: {dest_mask_full_path}")
                    copied_mask_count -=1
                except OSError as e:
                    print(f"Error cleaning up mask {dest_mask_full_path}: {e}")
    
    print(f"\nProcessed {actual_rows_to_process} rows from the original CSV.")
    if valid_rows_data:
        new_df = pd.DataFrame(valid_rows_data)
        new_df.to_csv(subset_csv_path, index=False)
        print(f"\nSubset dataset creation summary:")
        print(f"  New CSV created: {subset_csv_path} with {len(new_df)} valid rows.")
        print(f"  Successfully copied {copied_image_count} images to {subset_images_dir}")
        print(f"  Successfully copied {copied_mask_count} masks to {subset_masks_dir}")
        print(f"  Total samples skipped due to issues: {skipped_count}")
    else:
        print("\nNo valid samples were processed to create the subset dataset.")
        # Attempt to remove the subset CSV if it was created but is empty (though logic above should prevent empty CSV)
        if os.path.exists(subset_csv_path) and (not valid_rows_data or os.path.getsize(subset_csv_path) == 0) :
            try:
                os.remove(subset_csv_path)
                print(f"Removed empty or unnecessary subset CSV: {subset_csv_path}")
            except OSError as e:
                print(f"Error removing subset CSV {subset_csv_path}: {e}")


if __name__ == "__main__":
    # This script assumes it's located in the MedReSeg project root directory.
    project_root_dir = os.path.dirname(os.path.abspath(__file__)) 
    
    # Define paths relative to the project root
    original_csv = os.path.join(project_root_dir, 'dataset', 'SAMed2Dv1', 'SAMed2D_image_metadata_per_mask_with_questions.csv')
    original_data_root = os.path.join(project_root_dir, 'dataset', 'SAMed2Dv1') # Base for original images/masks
    
    # Destination for the subset
    subset_destination_base = os.path.join(project_root_dir, 'dataset_subset', 'SAMed2Dv1')
    
    number_of_rows_to_subset = 100 # You can change this value
    
    print(f"Starting dataset subset creation process...")
    print(f"  Original CSV: {original_csv}")
    print(f"  Original Data Root: {original_data_root}")
    print(f"  Subset Destination: {subset_destination_base}")
    print(f"  Number of rows to process: {number_of_rows_to_subset}")
    print("-" * 30)

    create_subset(original_csv, original_data_root, subset_destination_base, num_rows=number_of_rows_to_subset)
    
    print("-" * 30)
    print("Dataset subset creation process finished.")
