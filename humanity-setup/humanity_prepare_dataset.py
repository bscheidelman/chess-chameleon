import chess
import chess.engine
import h5py
import numpy as np
import logging
import os
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# --- IMPORTANT: UPDATE THIS PATH ---
STOCKFISH_PATH = "/Users/benscheidelman/Documents/stockfish-macos-m1-apple-silicon"
SOURCE_H5 = "../stylistic-setup/london_system_dataset.h5"
OUTPUT_H5 = "humanity_dataset_optimized.h5"

# Analysis Parameters
ENGINE_DEPTH = 12
NORMALIZATION_CAP = 500
# Increased batch size for less frequent I/O
WRITE_BATCH_SIZE = 8192 
# Target the number of performance cores on modern CPUs like Apple Silicon
# For your M4 Pro (8 performance cores), 8 is a great starting point.
NUM_PROCESSES = 8

# --- Global Engine for Multiprocessing Workers ---
# This will hold the engine instance for each worker process
worker_engine = None

def init_worker():
    """
    Initializer for the multiprocessing.Pool.
    Creates one persistent engine instance per worker process.
    This is the KEY to the performance improvement.
    """
    global worker_engine
    logging.info(f"Initializing engine for worker process {os.getpid()}...")
    try:
        worker_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        # Each worker is a separate process, so it can use 1 thread without conflicting.
        worker_engine.configure({"Threads": 1})
    except Exception as e:
        logging.critical(f"Worker {os.getpid()} failed to initialize engine: {e}")
        # Propagate exception to stop the pool
        raise e

def close_worker():
    """Finalizer to clean up the engine when the pool closes."""
    global worker_engine
    if worker_engine:
        logging.info(f"Closing engine for worker process {os.getpid()}...")
        worker_engine.quit()

# --- Helper Functions ---
PIECE_MAP = {
    0: (chess.PAWN, chess.WHITE), 1: (chess.KNIGHT, chess.WHITE),
    2: (chess.BISHOP, chess.WHITE), 3: (chess.ROOK, chess.WHITE),
    4: (chess.QUEEN, chess.WHITE), 5: (chess.KING, chess.WHITE),
    6: (chess.PAWN, chess.BLACK), 7: (chess.KNIGHT, chess.BLACK),
    8: (chess.BISHOP, chess.BLACK), 9: (chess.ROOK, chess.BLACK),
    10: (chess.QUEEN, chess.BLACK), 11: (chess.KING, chess.BLACK),
}

def tensor_to_board(tensor):
    """Converts our 16-channel tensor back into a python-chess Board object."""
    board = chess.Board(fen=None)
    for channel, (piece_type, color) in PIECE_MAP.items():
        for rank in range(8):
            for file in range(8):
                if tensor[channel, rank, file] == 1:
                    board.set_piece_at(chess.square(file, rank), chess.Piece(piece_type, color))
    
    board.turn = chess.WHITE if tensor[13, 0, 0] == 1.0 else chess.BLACK
    
    cas_rights = ""
    if tensor[12, 0, 7] == 1: cas_rights += "K"
    if tensor[12, 0, 0] == 1: cas_rights += "Q"
    if tensor[12, 7, 7] == 1: cas_rights += "k"
    if tensor[12, 7, 0] == 1: cas_rights += "q"
    board.set_castling_fen(cas_rights if cas_rights else "-")

    ep_squares = np.where(tensor[14] == 1.0)
    if len(ep_squares[0]) > 0:
        board.ep_square = chess.square(ep_squares[1][0], ep_squares[0][0])
        
    return board

# --- Multiprocessing Worker Function ---
def analyze_position(args):
    """
    Worker function to analyze a single position with Stockfish.
    Crucially, this now uses the pre-initialized 'worker_engine'.
    """
    global worker_engine
    if worker_engine is None:
        # This should not happen if the initializer worked correctly
        return args[0], -2.0 # Use a different error code for debugging

    index, input_tensor, human_move_int = args
    
    try:
        board = tensor_to_board(input_tensor)
        
        # 1. Get engine's evaluation of the position's best move
        info = worker_engine.analyse(board, chess.engine.Limit(depth=ENGINE_DEPTH))
        engine_score_obj = info['score'].white()
        
        # 2. Get engine's evaluation of the move the human actually played
        from_square = human_move_int // 64
        to_square = human_move_int % 64
        human_move = chess.Move(from_square, to_square)
        
        # If move is illegal, it's the worst possible move. Score it as 1.0.
        if human_move not in board.legal_moves:
            return index, 1.0 

        human_info = worker_engine.analyse(board, chess.engine.Limit(depth=ENGINE_DEPTH), root_moves=[human_move])
        human_score_obj = human_info['score'].white()

        # 3. Calculate the centipawn delta
        # If either side has a forced mate, the centipawn delta isn't meaningful.
        # A move that doesn't see a mate could be a huge blunder.
        # Capping delta in mate scenarios.
        if engine_score_obj.is_mate() or human_score_obj.is_mate():
            engine_mate = engine_score_obj.mate() or 0
            human_mate = human_score_obj.mate() or 0
            # If engine had mate and human move doesn't, it's a blunder.
            if engine_mate > 0 and human_mate <= 0:
                 delta = NORMALIZATION_CAP
            # If engine had no mate but human move allows one, it's a blunder.
            elif engine_mate <= 0 and human_mate < 0:
                 delta = NORMALIZATION_CAP
            else:
                 delta = 0
        else:
            engine_cp = engine_score_obj.score()
            human_cp = human_score_obj.score()
            delta = abs(engine_cp - human_cp) if engine_cp is not None and human_cp is not None else 0

        # 4. Normalize to the Humanity Score
        humanity_score = min(delta / NORMALIZATION_CAP, 1.0)
        
        return index, humanity_score

    except Exception as e:
        logging.error(f"Error analyzing position {index} in worker {os.getpid()}: {e}")
        return index, -1.0


def get_rough_eta(total_positions_to_run):
    """
    Runs a small batch of analyses to provide a more stable ETA.
    """
    logging.info("Performing a dry run on 100 positions to estimate total time...")
    with h5py.File(SOURCE_H5, 'r') as f:
        sample_args_list = [(i, f['inputs'][i], f['outputs'][i]) for i in range(100)]
    
    init_worker() # Manually init an engine for this test
    start_time = time.time()
    for args in sample_args_list:
        analyze_position(args)
    end_time = time.time()
    close_worker() # Clean up the test engine

    time_per_pos = (end_time - start_time) / 100
    total_eta_seconds = time_per_pos * total_positions_to_run
    
    if total_eta_seconds > 0:
      eta_str = time.strftime('%H hours, %M minutes, %S seconds', time.gmtime(total_eta_seconds))
      logging.info(f"Time per position: {time_per_pos:.4f} seconds.")
      logging.info(f"Estimated time for remaining dataset: {eta_str}")
    else:
      logging.info("ETA calculation resulted in zero; execution may be extremely fast.")


# --- Main Orchestration Function ---
def main():
    if not os.path.exists(STOCKFISH_PATH) or not os.path.isfile(STOCKFISH_PATH):
        logging.critical(f"FATAL: Stockfish engine not found at '{STOCKFISH_PATH}'.")
        return

    if not os.path.exists(SOURCE_H5):
        logging.critical(f"FATAL: Source file '{SOURCE_H5}' not found.")
        return

    with h5py.File(SOURCE_H5, 'r') as source_f:
        total_positions = len(source_f['inputs'])
        
        start_index = 0
        if os.path.exists(OUTPUT_H5):
            logging.info(f"Output file '{OUTPUT_H5}' exists. Checking for resume point.")
            with h5py.File(OUTPUT_H5, 'r') as f:
                outputs_data = f['outputs'][:]
                # Find the first index that is still the fillvalue (-1.0)
                non_processed = np.where(outputs_data == -1.0)[0]
                start_index = non_processed[0] if len(non_processed) > 0 else total_positions
            logging.info(f"Resuming from position #{start_index}")
        else:
            with h5py.File(OUTPUT_H5, 'w') as f:
                f.create_dataset('inputs', data=source_f['inputs'][:])
                f.create_dataset('outputs', (total_positions,), dtype='float32', fillvalue=-1.0)

        if start_index >= total_positions:
            logging.info("Dataset preparation is already complete.")
            return

        tasks_to_run = total_positions - start_index
        get_rough_eta(tasks_to_run)

        def task_generator():
            with h5py.File(SOURCE_H5, 'r') as f_source:
                for i in range(start_index, total_positions):
                    yield (i, f_source['inputs'][i], f_source['outputs'][i])
        
        logging.info(f"Starting analysis on {tasks_to_run} positions with {NUM_PROCESSES} worker processes.")
        
        results_buffer = []
        running_avg_score = 0
        processed_count = 0

        # Calculate a dynamic chunksize to balance overhead and memory.
        # This creates a decent number of chunks per worker.
        chunksize = max(1, min(128, (tasks_to_run // NUM_PROCESSES) // 4))
        logging.info(f"Using a task chunksize of {chunksize}")

        with Pool(processes=NUM_PROCESSES, initializer=init_worker) as pool, h5py.File(OUTPUT_H5, 'a') as output_f:
            outputs_dset = output_f['outputs']
            progress_bar = tqdm(pool.imap_unordered(analyze_position, task_generator(), chunksize=chunksize), total=tasks_to_run, desc="Analyzing Positions")
            
            for index, humanity_score in progress_bar:
                if humanity_score >= 0: # Exclude errors
                    results_buffer.append((index, humanity_score))
                    processed_count += 1
                    running_avg_score += (humanity_score - running_avg_score) / processed_count

                if len(results_buffer) >= WRITE_BATCH_SIZE:
                    for idx, score in results_buffer:
                        outputs_dset[idx] = score
                    output_f.flush() # Ensure data is written to disk
                    results_buffer = []

                progress_bar.set_postfix(avg_score=f"{running_avg_score:.3f}")

            if results_buffer:
                for idx, score in results_buffer:
                    outputs_dset[idx] = score
                output_f.flush()

    logging.info("Humanity Dataset preparation complete.")

if __name__ == "__main__":
    main()

