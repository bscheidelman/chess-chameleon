import chess
import chess.pgn
import numpy as np
import h5py
import time
import logging
import re
import io
import os
import argparse
import zstandard as zstd
from multiprocessing import Pool, cpu_count

#download db's from https://database.lichess.org/

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# The path to the large, Zstandard-compressed PGN database file (.pgn.zst) from Lichess.
# THIS CONSTANT HAS BEEN REMOVED TO ALLOW FOR MULTI-FILE PROCESSING IN main()
# PGN_FILE_PATH = "../data/lichess_db_standard_rated_2016-03.pgn.zst" 

MIN_ELO_RATING = 2200  # Player rating is now a fixed constant.

# --- Piece to Channel Mapping ---
PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}
NUM_CHANNELS = 16

# --- Helper Functions ---
def board_to_tensor(board):
    """Converts a chess.Board object into a (16, 8, 8) tensor representation."""
    tensor = np.zeros((NUM_CHANNELS, 8, 8), dtype=np.float32)
    
    # Piece placement (Channels 0-11)
    for square, piece in board.piece_map().items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
        tensor[channel, rank, file] = 1
        
    # Castling rights (Channel 12)
    if board.has_kingside_castling_rights(chess.WHITE): tensor[12, 0, 7] = 1
    if board.has_queenside_castling_rights(chess.WHITE): tensor[12, 0, 0] = 1
    if board.has_kingside_castling_rights(chess.BLACK): tensor[12, 7, 7] = 1
    if board.has_queenside_castling_rights(chess.BLACK): tensor[12, 7, 0] = 1
    
    # Side to move (Channel 13 - entire plane)
    tensor[13, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    
    # En passant square (Channel 14)
    if board.ep_square:
        rank, file = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        tensor[14, rank, file] = 1.0
        
    # Move number (Channel 15 - entire plane, scaled)
    tensor[15, :, :] = board.fullmove_number / 100.0
    return tensor

def move_to_policy_target(move):
    """Converts a move object into a 0-4095 integer target."""
    return move.from_square * 64 + move.to_square

# --- Multiprocessing Worker Function ---
def process_game_chunk(pgn_text):
    """
    Processes a single game's PGN text for ELO and opening filtering, 
    and converts the positions/moves to targets if successful.
    """
    
    # --- Filter 1: Check ELO with regex ---
    elo_regex = re.compile(r"\[(WhiteElo|BlackElo)\s\"(\d+)\"\]")
    elos = dict(elo_regex.findall(pgn_text))
    white_elo = int(elos.get("WhiteElo", 0))
    black_elo = int(elos.get("BlackElo", 0))
    if (white_elo + black_elo) / 2 < MIN_ELO_RATING:
        return 'ELO_FAIL', None

    # --- Filter 2: London System Opening text search (1. d4 followed by Bf4) ---
    movetext_start_index = pgn_text.find("\n\n1.")
    if movetext_start_index == -1:
        return 'TEXT_FAIL', None
    
    movetext_header = pgn_text[movetext_start_index : movetext_start_index + 80]
    pattern = re.compile(r"1\.\s*d4.*Bf4")
    if not pattern.search(movetext_header):
        return 'TEXT_FAIL', None
    
    # --- Final Step: Parse the validated game for tensor conversion ---
    try:
        # Use io.StringIO for in-memory PGN reading
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return 'PARSE_FAIL', None
    except Exception:
        return 'PARSE_ERROR', None

    # --- Process the valid game, collecting white moves only ---
    board = game.board()
    game_positions = []
    game_moves = []
    for move in game.mainline_moves():
        if board.turn == chess.WHITE:
            game_positions.append(board_to_tensor(board))
            game_moves.append(move_to_policy_target(move))
        board.push(move)
    
    if game_positions:
        return 'SUCCESS', {"inputs": game_positions, "outputs": game_moves}
    
    return 'NO_POSITIONS', None

def initialize_hdf5_file(output_file):
    """Initializes the HDF5 file with resizable datasets."""
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('inputs', (0, NUM_CHANNELS, 8, 8), maxshape=(None, NUM_CHANNELS, 8, 8), dtype='float32', chunks=True, compression="gzip")
        f.create_dataset('outputs', (0,), maxshape=(None,), dtype='int32', chunks=True, compression="gzip")
    logging.info(f"HDF5 file '{output_file}' initialized with gzip compression.")

def append_to_hdf5(output_file, inputs, outputs):
    """Appends new data to the existing HDF5 datasets."""
    if not inputs:
        return
    with h5py.File(output_file, 'a') as f:
        input_dset = f['inputs']
        input_dset.resize(input_dset.shape[0] + len(inputs), axis=0)
        input_dset[-len(inputs):] = inputs
        output_dset = f['outputs']
        output_dset.resize(output_dset.shape[0] + len(outputs), axis=0)
        output_dset[-len(outputs):] = outputs

def print_progress(games_found, positions, percent, eta_seconds, scanned_count):
    """Prints a single line progress report, updating in place."""
    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
    progress_bar = f"[{'=' * int(percent // 2)}{' ' * (50 - int(percent // 2))}]"
    print(f"Progress: {progress_bar} {percent:.2f}% | Scanned: {scanned_count} | Found: {games_found} | Positions: {positions} | ETA: {eta_str}", end='\r')

def main(args):
    # --- New: Define the list of files to process ---
    PGN_FILES = [
        "/Users/benscheidelman/Documents/All Projects/Chess Humanity/data/lichess_db_standard_rated_2016-03.pgn.zst",
        "/Users/benscheidelman/Documents/All Projects/Chess Humanity/data/lichess_db_standard_rated_2016-04.pgn.zst",
        "/Users/benscheidelman/Documents/All Projects/Chess Humanity/data/lichess_db_standard_rated_2016-05.pgn.zst",
        "/Users/benscheidelman/Documents/All Projects/Chess Humanity/data/lichess_db_standard_rated_2016-06.pgn.zst",
        "/Users/benscheidelman/Documents/All Projects/Chess Humanity/data/lichess_db_standard_rated_2016-07.pgn.zst",
        "/Users/benscheidelman/Documents/All Projects/Chess Humanity/data/lichess_db_standard_rated_2016-08.pgn.zst",
        "/Users/benscheidelman/Documents/All Projects/Chess Humanity/data/lichess_db_standard_rated_2016-09.pgn.zst",
        "/Users/benscheidelman/Documents/All Projects/Chess Humanity/data/lichess_db_standard_rated_2016-10.pgn.zst",
        "/Users/benscheidelman/Documents/All Projects/Chess Humanity/data/lichess_db_standard_rated_2016-11.pgn.zst",
        "/Users/benscheidelman/Documents/All Projects/Chess Humanity/data/lichess_db_standard_rated_2016-12.pgn.zst",
    ]

    # --- HDF5 File Initialization ---
    if not os.path.exists(args.output_file):
        initialize_hdf5_file(args.output_file)
    else:
        logging.info(f"Output file '{args.output_file}' already exists. Will append new data.")
    
    # Global tracking for statistics across all files
    global_start_time = time.time()
    global_stats = {
        'SUCCESS': 0, 'ELO_FAIL': 0, 'TEXT_FAIL': 0, 'PARSE_FAIL': 0, 
        'PARSE_ERROR': 0, 'NO_POSITIONS': 0, 'SCANNED': 0
    }
    total_positions_collected = 0
    
    num_processes = max(1, cpu_count() - 1)
    logging.info(f"Starting processing with {num_processes} worker processes.")

    # --- Loop through each file ---
    for i, pgn_file_path in enumerate(PGN_FILES):
        logging.info(f"\n--- Starting File {i+1}/{len(PGN_FILES)}: {os.path.basename(pgn_file_path)} ---")
        
        # Resume logic reset for clarity in multi-file processing
        resume_byte_offset = 0 
        
        try:
            total_file_size = os.path.getsize(pgn_file_path)
        except FileNotFoundError:
            logging.critical(f"ERROR: PGN file not found at '{pgn_file_path}'. Skipping.")
            continue
            
        file_start_time = time.time()

        try:
            with open(pgn_file_path, 'rb') as f, Pool(processes=num_processes) as pool:
                if resume_byte_offset > 0:
                    # Not currently supported in multi-file but kept structure for single file resume potential
                    f.seek(resume_byte_offset)
                
                dctx = zstd.ZstdDecompressor()
                stream_reader = dctx.stream_reader(f)
                text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
                
                def game_generator():
                    """Generator that yields one complete PGN game at a time."""
                    game_text = ""
                    for line in text_stream:
                        if line.startswith("[Event") and game_text:
                            yield game_text
                            game_text = line
                        else:
                            game_text += line
                    if game_text:
                        yield game_text

                logging.info("Starting to iterate through game generator with worker pool...")
                chunk_size = num_processes * 1000
                
                for status, result in pool.imap_unordered(process_game_chunk, game_generator(), chunksize=chunk_size):
                    global_stats['SCANNED'] += 1
                    global_stats[status] = global_stats.get(status, 0) + 1

                    if status == 'SUCCESS' and result:
                        num_new_positions = len(result['inputs'])
                        total_positions_collected += num_new_positions
                        append_to_hdf5(args.output_file, result['inputs'], result['outputs'])

                    if global_stats['SCANNED'] % 2000 == 0:
                        current_pos = f.tell()
                        elapsed_time = time.time() - file_start_time
                        bytes_processed = current_pos - resume_byte_offset
                        speed_bytes_sec = bytes_processed / elapsed_time if elapsed_time > 0 else 0
                        
                        # Use file-specific progress for accurate ETA
                        percent_complete = current_pos / total_file_size * 100
                        eta = (total_file_size - current_pos) / speed_bytes_sec if speed_bytes_sec > 0 else 0
                        
                        # Calculate current positions in HDF5 file
                        current_hdf5_positions = 0
                        try:
                            with h5py.File(args.output_file, 'r') as f_h5:
                                current_hdf5_positions = f_h5['outputs'].shape[0]
                        except Exception:
                            pass # Ignore if file read fails during progress update

                        print_progress(global_stats['SUCCESS'], current_hdf5_positions, percent_complete, eta, global_stats['SCANNED'])
                
                # --- File finished: Ensure no single-file resume data remains ---
                final_byte_pos = f.tell()
                with h5py.File(args.output_file, 'a') as f_h5:
                    if 'last_byte_offset' in f_h5.attrs:
                         del f_h5.attrs['last_byte_offset']
                logging.info(f"\nCompleted file: {os.path.basename(pgn_file_path)}. Processed until byte: {final_byte_pos}")

        except Exception as e:
            logging.critical(f"A critical unexpected error occurred during processing file {pgn_file_path}: {e}", exc_info=True)

    # --- Final Summary across all files ---
    print("\n\n--- Data Collection Complete (All Files Processed) ---")
    print(f"Total Games Scanned: {global_stats['SCANNED']}")
    print("Filter Results:")
    print(f"  - Passed (SUCCESS):      {global_stats['SUCCESS']}")
    print(f"  - Rejected (Low ELO):    {global_stats['ELO_FAIL']}")
    print(f"  - Rejected (Text Search):{global_stats['TEXT_FAIL']}")
    print(f"  - Rejected (Parse Fail): {global_stats['PARSE_FAIL'] + global_stats['PARSE_ERROR']}")
    print("-" * 30)
    print(f"Total London System games collected: {global_stats['SUCCESS']}")
    
    # Read final position count directly from HDF5
    final_position_count = 0
    try:
        with h5py.File(args.output_file, 'r') as f_h5:
            final_position_count = f_h5['outputs'].shape[0]
    except Exception:
        pass

    print(f"Total board positions stored: {final_position_count}")
    print(f"Dataset saved to '{args.output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Lichess PGN databases to extract London System games.")
    parser.add_argument("-o", "--output_file", default="new_london_system_dataset.h5", help="Path to the output HDF5 file.")
    
    args = parser.parse_args()
    main(args)
