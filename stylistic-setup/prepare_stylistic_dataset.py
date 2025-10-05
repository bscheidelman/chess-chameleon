import h5py
import numpy as np
import random
import logging
import os
import time
import multiprocessing

# --- Setup Logging ---
# Note: In multiprocessing, logging from workers might interleave, but it's essential for tracking progress.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configuration ---
SOURCE_H5 = "london_system_dataset.h5"
SPLIT_RATIOS = {'train': 0.85, 'validation': 0.10, 'test': 0.05}

def print_progress(current, total, start_time, prefix=''):
    """Helper function to display a progress bar with ETA."""
    if current == 0:
        current = 1
    elapsed_time = time.time() - start_time
    speed = current / elapsed_time
    percent = 100 * (current / float(total))
    remaining = total - current
    eta = remaining / speed if speed > 0 else 0
    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
    
    # Ensure prefix is consistent length for cleaner terminal output
    full_prefix = f"{prefix:<25}"
    
    # 50-character progress bar
    progress_bar = f"[{'=' * int(percent // 2):<50}]"
    
    # Use f-string to manage output length and flush to stdout
    print(f"{full_prefix} {progress_bar} {percent:6.2f}% | {current:8}/{total:8} | ETA: {eta_str}", end='\r', flush=True)
    
    if current == total:
        print(f"{full_prefix} [{'=' * 50}] 100.00% | {total:8}/{total:8} | Complete.             ", flush=True)

def write_split_file(args):
    """
    Worker function to handle the creation and writing of a single HDF5 split file.
    Opens its own handle to the source file to ensure process isolation.
    """
    source_file_path, split_name, sorted_indices = args
    
    file_name = f"{split_name}.h5"
    total_to_write = len(sorted_indices)
    
    logging.info(f"[Worker {split_name.upper()}] Starting write to {file_name} with {total_to_write} samples.")
    
    try:
        # Open source file within the worker process
        with h5py.File(source_file_path, 'r') as source_f:
            
            with h5py.File(file_name, 'w') as dest_f:
                # Determine shapes and dtypes dynamically from source (assuming they are consistent)
                input_shape = (total_to_write,) + source_f['inputs'].shape[1:]
                output_shape = (total_to_write,)

                inputs_dset = dest_f.create_dataset('inputs', input_shape, dtype='float32', compression="gzip")
                outputs_dset = dest_f.create_dataset('outputs', output_shape, dtype='int32', compression="gzip")
                
                start_time = time.time()
                chunk_size = 4096  # Increased chunk size for better I/O performance
                written_count = 0

                for i in range(0, total_to_write, chunk_size):
                    chunk_indices = sorted_indices[i : i + chunk_size]
                    
                    # Read a chunk from the source file using the sorted indices
                    input_chunk = source_f['inputs'][chunk_indices, ...]
                    output_chunk = source_f['outputs'][chunk_indices]
                    
                    # Write the entire chunk to the destination file
                    dest_start = i
                    dest_end = i + len(chunk_indices)
                    inputs_dset[dest_start:dest_end] = input_chunk
                    outputs_dset[dest_start:dest_end] = output_chunk

                    written_count += len(chunk_indices)
                    
                    # Update progress bar
                    print_progress(written_count, total_to_write, start_time, prefix=f"Writing {file_name}")
        
        logging.info(f"[Worker {split_name.upper()}] Successfully finished writing {file_name}.")
        
    except Exception as e:
        logging.error(f"[Worker {split_name.upper()}] Error during file writing: {e}")


def prepare_dataset():
    """
    Reads the original HDF5 dataset, balances it by game phase, and splits it
    into training, validation, and test sets using parallel processing for I/O.
    """
    if not os.path.exists(SOURCE_H5):
        logging.critical(f"ERROR: Source file '{SOURCE_H5}' not found. Please run the data collector first.")
        return

    logging.info(f"Opening source dataset: {SOURCE_H5}")
    # We only open the source file once here to get total samples and phase data
    with h5py.File(SOURCE_H5, 'r') as source_f:
        total_samples = len(source_f['inputs'])
        logging.info(f"Found {total_samples} total positions.")

        # --- 1. Categorize all positions by game phase (Sequential - fast CPU task) ---
        logging.info("Categorizing positions by game phase...")
        indices_by_phase = {'opening': [], 'middlegame': [], 'endgame': []}
        
        # Load the move number channel into memory for faster processing
        # Ensure array slicing is safe
        if source_f['inputs'].shape[1] > 15:
            move_number_channel = source_f['inputs'][:, 15, 0, 0] * 100
        else:
            logging.error("Input data does not have the expected 16 channels for move number. Aborting.")
            return

        for i, move_num in enumerate(move_number_channel):
            if move_num <= 12:
                indices_by_phase['opening'].append(i)
            elif move_num <= 40:
                indices_by_phase['middlegame'].append(i)
            else:
                indices_by_phase['endgame'].append(i)
        
        logging.info(f"Categorized positions: {len(indices_by_phase['opening'])} opening, "
                     f"{len(indices_by_phase['middlegame'])} middlegame, "
                     f"{len(indices_by_phase['endgame'])} endgame.")

        # --- 2. Create balanced lists of indices for each split (Sequential - fast CPU task) ---
        logging.info("Splitting indices into train, validation, and test sets...")
        split_indices_map = {'train': [], 'validation': [], 'test': []}
        for phase, indices in indices_by_phase.items():
            random.shuffle(indices)
            
            train_end = int(len(indices) * SPLIT_RATIOS['train'])
            val_end = train_end + int(len(indices) * SPLIT_RATIOS['validation'])

            split_indices_map['train'].extend(indices[:train_end])
            split_indices_map['validation'].extend(indices[train_end:val_end])
            split_indices_map['test'].extend(indices[val_end:])

    
    # --- 3. Parallelize the writing (I/O-bound task) ---
    tasks_to_run = []
    
    # Prepare arguments for each worker process
    for split_name, indices in split_indices_map.items():
        # Sorting is performed here in the main thread before passing, 
        # optimizing read patterns in the worker (though adding a slight delay upfront).
        logging.info(f"  Optimizing read order for {split_name} set ({len(indices)} samples)...")
        sorted_indices = sorted(indices)
        
        # Args: (source_path, split_name, sorted_indices)
        tasks_to_run.append((SOURCE_H5, split_name, sorted_indices))
        
    logging.info("Index preparation complete. Starting parallel writing...")

    # Use a Pool with 3 processes, one dedicated to writing each file.
    # This overlaps the I/O time for the validation and test sets with the large train set write.
    # Note: We do not pass the h5py.File object (source_f) directly, but the path, 
    # forcing each worker to open its own safe read connection.
    try:
        with multiprocessing.Pool(processes=3) as pool:
            # Map the write_split_file function to the list of tasks
            pool.map(write_split_file, tasks_to_run)
            
        logging.info("All HDF5 files written successfully.")
    except Exception as e:
        logging.critical(f"A critical error occurred during parallel processing: {e}")


if __name__ == "__main__":
    # Ensure multiprocessing starts cleanly on all platforms
    multiprocessing.freeze_support() 
    prepare_dataset()
