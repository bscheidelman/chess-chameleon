import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import chess
import chess.svg
import chess.engine
from typing import Dict, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Configuration ---
# Make sure this path points to your actual Stockfish executable
STOCKFISH_PATH = "/Users/benscheidelman/Documents/stockfish-macos-m1-apple-silicon"
PERSONALITY_MODEL_PATH = "models/chess_imitator_final.pth"
HUMANITY_MODEL_PATH = "models/humanity_model_resnet_v5.pth"

STOCKFISH_DEPTH = 18
NUM_CHANNELS = 16

# --- Piece to Channel Mapping ---
PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}

# ==============================================================================
# --- MODEL ARCHITECTURES (COPIED FROM ORIGINAL SCRIPT) ---
# ==============================================================================

# --- Personality Model Architecture ---
class PersonalityResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PersonalityResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ImitationResNet(nn.Module):
    def __init__(self, in_channels=16, num_blocks=8, num_filters=128):
        super(ImitationResNet, self).__init__()
        self.initial_conv = nn.Sequential(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1), nn.BatchNorm2d(num_filters), nn.ReLU(inplace=True))
        self.residual_tower = nn.Sequential(*[PersonalityResidualBlock(num_filters, num_filters) for _ in range(num_blocks)])
        self.from_head = nn.Sequential(nn.Conv2d(num_filters, 2, kernel_size=1), nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(), nn.Linear(2 * 8 * 8, 64))
        self.to_head = nn.Sequential(nn.Conv2d(num_filters, 2, kernel_size=1), nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(), nn.Linear(2 * 8 * 8, 64))
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.residual_tower(x)
        return self.from_head(x), self.to_head(x)

# --- Humanity Model Architecture ---
class HumanityResidualBlock(nn.Module):
    def __init__(self, channels):
        super(HumanityResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class HumanityResNet(nn.Module):
    def __init__(self, in_channels=16, num_blocks=8, num_filters=128):
        super(HumanityResNet, self).__init__()
        self.initial_conv = nn.Sequential(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1), nn.BatchNorm2d(num_filters), nn.ReLU(inplace=True))
        self.residual_tower = nn.Sequential(*[HumanityResidualBlock(num_filters) for _ in range(num_blocks)])
        self.regression_head = nn.Sequential(nn.Conv2d(num_filters, 4, kernel_size=1), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 8 * 8, 256), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 1), nn.Sigmoid())
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.residual_tower(x)
        return self.regression_head(x)

# ==============================================================================
# --- Helper Functions and Model Loading ---
# ==============================================================================

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = np.zeros((NUM_CHANNELS, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
        tensor[channel, rank, file] = 1
    if board.has_kingside_castling_rights(chess.WHITE): tensor[12, 0, 7] = 1
    if board.has_queenside_castling_rights(chess.WHITE): tensor[12, 0, 0] = 1
    if board.has_kingside_castling_rights(chess.BLACK): tensor[12, 7, 7] = 1
    if board.has_queenside_castling_rights(chess.BLACK): tensor[12, 7, 0] = 1
    tensor[13, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    if board.ep_square:
        rank, file = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        tensor[14, rank, file] = 1.0
    tensor[15, :, :] = board.fullmove_number / 100.0
    return torch.from_numpy(tensor).unsqueeze(0)

# Load models into global scope for efficiency
try:
    personality_model = ImitationResNet()
    personality_model.load_state_dict(torch.load(PERSONALITY_MODEL_PATH, map_location=torch.device('cpu')))
    personality_model.eval()

    humanity_model = HumanityResNet()
    humanity_model.load_state_dict(torch.load(HUMANITY_MODEL_PATH, map_location=torch.device('cpu')))
    humanity_model.eval()
except FileNotFoundError as e:
    print(f"ERROR: Could not find model files. Make sure .pth files are in the correct /models directory. Error: {e}")
    exit()

def get_stockfish_weights(fen: str) -> Dict[str, float]:
    if not os.path.exists(STOCKFISH_PATH):
        print(f"Stockfish engine not found at '{STOCKFISH_PATH}'")
        return None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except FileNotFoundError:
        print(f"Could not start the engine at {STOCKFISH_PATH}.")
        return None
    
    board = chess.Board(fen)
    if board.is_game_over():
        engine.quit()
        return {}

    move_evals = {}
    analysis_limit = chess.engine.Limit(depth=STOCKFISH_DEPTH)
    
    infos = engine.analyse(board, analysis_limit, multipv=len(list(board.legal_moves)))
    
    for info in infos:
        move = info.get("pv")[0]
        if move:
            score = info["score"].relative.score(mate_score=10000)
            final_score = score if board.turn == chess.WHITE else -score
            move_san = board.san(move)
            move_evals[move_san] = final_score / 100.0
            
    engine.quit()
    return move_evals

def get_engine_move(fen: str) -> Tuple[chess.Move, pd.DataFrame, float]:
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, {}, 0.0
    
    board_tensor = board_to_tensor(board)

    with torch.no_grad():
        humanity_score = humanity_model(board_tensor).item()
        from_logits, to_logits = personality_model(board_tensor)
        from_probs = torch.softmax(from_logits, dim=1).squeeze(0)
        to_probs = torch.softmax(to_logits, dim=1).squeeze(0)

        raw_style_scores = {}
        for move in legal_moves:
            p_from = from_probs[move.from_square].item()
            p_to = to_probs[move.to_square].item()
            raw_style_scores[board.san(move)] = p_from * p_to

        total_style_score = sum(raw_style_scores.values())
        style_scores = {san: score / total_style_score if total_style_score > 0 else 0 
                        for san, score in raw_style_scores.items()}

    tactic_scores_raw = get_stockfish_weights(fen)
    


    if humanity_score <= 0.4:
        w_style = 0.2
    else:
        x_scaled = (humanity_score - 0.4) / (1.0 - 0.4)
        s = 3 * x_scaled**2 - 2 * x_scaled**3
        w_style = 0.2 + s * (0.85 - 0.2)

    w_tactic = 1 - w_style
    
    if not tactic_scores_raw:
         return legal_moves[0], {}, humanity_score

    OPTIMAL_TAU = 0.40
    tactic_values = np.array([tactic_scores_raw.get(board.san(m), 0) for m in legal_moves])
    
    
    exp_scores = np.exp(tactic_values / OPTIMAL_TAU)
    softmax_scores = exp_scores / np.sum(exp_scores)
    transformed_tactic_scores = {board.san(m): s for m, s in zip(legal_moves, softmax_scores)}

    best_move = None
    max_score = -float('inf')
    best_move_style_score = 0
    best_move_tactic_score = 0

    for move in legal_moves:
        san = board.san(move)
        style = style_scores.get(san, 0)
        transformed_tactic = transformed_tactic_scores.get(san, 0)
        final_score = (w_style * style) + (w_tactic * transformed_tactic)

        if final_score > max_score:
            max_score = final_score
            best_move = move
            best_move_style_score = style
            best_move_tactic_score = transformed_tactic
    
    analysis_data = {
        "humanity": humanity_score,
        "style_score": best_move_style_score,
        "tactic_score": best_move_tactic_score,
    }



    return best_move, analysis_data

# ==============================================================================
# --- Flask API Server ---
# ==============================================================================
app = Flask(__name__)
CORS(app)

@app.route('/api/get-move', methods=['POST'])
def get_move_api():
    data = request.get_json()
    fen = data.get('fen')
    if not fen:
        return jsonify({"error": "FEN string not provided"}), 400

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "Invalid FEN string"}), 400

    if board.is_game_over():
        return jsonify({"best_move": None, "analysis": {}, "message": "Game is over."})

    best_move, analysis = get_engine_move(fen)

    if best_move:
        return jsonify({
            "best_move": best_move.uci(),
            "analysis": analysis
        })
    else:
        return jsonify({"error": "Could not determine a best move."}), 500

if __name__ == '__main__':
    app.run(debug=True)

