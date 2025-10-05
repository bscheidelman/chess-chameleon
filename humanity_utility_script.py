import h5py
import numpy as np
import logging
import os
from tqdm import tqdm
import contextlib

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
SOURCE_H5 = "humanity_dataset_optimized.h5"
TRAIN_H5_OUT = "train_humanity.h5"
VALIDATION_H5_OUT = "validation_humanity.h5"
TEST_H5_OUT = "test_humanity.h5"

SPLIT_RATIOS = {'train': 0.80, 'validation': 0.10, 'test': 0.10}
# Use a larger chunk size for fewer, more efficient I/O operations
CHUNK_SIZE = 16384 

def split_dataset_optimized():
    """
    Reads the complete humanity_dataset.h5 file sequentially and splits it into
    training, validation, and test sets in a single pass to maximize I/O efficiency.
    """
    if not os.path.exists(SOURCE_H5):
        logging.critical(f"ERROR: Source file '{SOURCE_H5}' not found. Please run humanity_prepare_dataset.py first.")
        return

    logging.info(f"Opening source dataset: {SOURCE_H5}")
    
    # Use contextlib.ExitStack to manage multiple file handles cleanly
    with contextlib.ExitStack() as stack:
        source_f = stack.enter_context(h5py.File(SOURCE_H5, 'r'))
        
        # Open all destination files for writing
        dest_files = {
            'train': stack.enter_context(h5py.File(TRAIN_H5_OUT, 'w')),
            'validation': stack.enter_context(h5py.File(VALIDATION_H5_OUT, 'w')),
            'test': stack.enter_context(h5py.File(TEST_H5_OUT, 'w'))
        }

        total_samples = len(source_f['inputs'])
        input_shape = source_f['inputs'].shape[1:]
        output_shape = source_f['outputs'].shape[1:]

        logging.info(f"Found {total_samples} total positions. Creating resizable output files.")

        # --- Create Resizable Datasets ---
        # We don't know the final size, so create resizable datasets.
        dest_datasets = {}
        for name, f in dest_files.items():
            dest_datasets[name] = {
                'inputs': f.create_dataset('inputs', (0, *input_shape), maxshape=(None, *input_shape), dtype='float32', compression="gzip"),
                'outputs': f.create_dataset('outputs', (0, *output_shape), maxshape=(None, *output_shape), dtype='float32', compression="gzip")
            }

        logging.info(f"Processing dataset in chunks of {CHUNK_SIZE}...")
        
        # --- Process the file in a single sequential pass ---
        progress_bar = tqdm(range(0, total_samples, CHUNK_SIZE), desc="Splitting dataset", unit="chunk")

        for i in progress_bar:
            # Read a large sequential chunk from the source
            chunk_start = i
            chunk_end = min(i + CHUNK_SIZE, total_samples)
            
            input_chunk = source_f['inputs'][chunk_start:chunk_end]
            output_chunk = source_f['outputs'][chunk_start:chunk_end]
            
            # --- In-memory assignment to splits ---
            # This is extremely fast
            num_in_chunk = len(input_chunk)
            assignments = np.random.choice(
                ['train', 'validation', 'test'], 
                size=num_in_chunk, 
                p=[SPLIT_RATIOS['train'], SPLIT_RATIOS['validation'], SPLIT_RATIOS['test']]
            )
            
            # --- Append data to destination files ---
            for split_name in dest_files.keys():
                mask = (assignments == split_name)
                count = np.sum(mask)
                
                if count == 0:
                    continue
                
                # Get the data for this split from the chunk
                inputs_to_write = input_chunk[mask]
                outputs_to_write = output_chunk[mask]
                
                # Resize the destination dataset
                dset_inputs = dest_datasets[split_name]['inputs']
                dset_outputs = dest_datasets[split_name]['outputs']
                
                old_size = dset_inputs.shape[0]
                dset_inputs.resize((old_size + count, *input_shape))
                dset_outputs.resize((old_size + count, *output_shape))
                
                # Write the new data
                dset_inputs[old_size:] = inputs_to_write
                dset_outputs[old_size:] = outputs_to_write

        logging.info("Dataset splitting complete. Final sizes:")
        for name, f in dest_files.items():
            final_size = len(f['inputs'])
            logging.info(f"  - {name} ({os.path.basename(f.filename)}): {final_size} samples")


if __name__ == "__main__":
    split_dataset_optimized()