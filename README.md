# The London AI: A Human-Styled Chess Engine

This project is the backend server for a unique chess AI that is specifically trained to play the London System. Unlike traditional chess engines that only seek the objectively best move, this AI blends tactical analysis with a learned "human" style, creating a more nuanced and interesting opponent.

The application is built in Python and uses a Flask web server to provide an API that a separate frontend (like the provided React `index.html` file) can communicate with.

## How It Works: The Three-Model System

The AI's decision-making process is a "negotiated" outcome between three distinct components:

1. **The Personality Model (`ImitationResNet`):** A deep neural network trained on a dataset of 3.7 million high-rated games where the London System was played. This model learns to predict which moves are most *characteristic* of a human London System player.

2. **The Humanity Model (`HumanityResNet`):** A second neural network that analyzes the complexity of a board position. It outputs a "humanity score," which is used to decide how much weight to give to stylistic moves versus purely tactical ones.

3. **The Tactical Engine (Stockfish):** The powerful and world-class Stockfish engine is used to provide an objective, tactical evaluation of all legal moves, ensuring the AI doesn't make simple blunders.

The final move is chosen by blending the outputs of the Personality Model and Stockfish, with the weights determined by the Humanity Score.

## Setup and Installation

Follow these steps to set up and run the server on your local machine.

### 1. Prerequisites

* Python 3.8+
* A virtual environment tool (like `venv`) is highly recommended.

### 2. Clone the Repository

Clone this project to your local machine.

```
git clone <your-repository-url>
cd <your-repository-directory>
```

### 3. Set Up the Environment

Create and activate a Python virtual environment.

```
# Create the virtual environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Activate it (Windows)
.\\venv\\Scripts\\activate
```

### 4. Install Python Dependencies

Install all the required Python libraries.

```
pip install torch numpy pandas python-chess Flask Flask-Cors
```

### 5. Download Stockfish

The AI requires the Stockfish engine for its tactical calculations.

1. Go to the official Stockfish download page: [**https://stockfishchess.org/download/**](https://stockfishchess.org/download/)
2. Download the appropriate version for your operating system (e.g., macOS Apple Silicon, Windows x64, etc.).
3. Unzip the downloaded file and place the Stockfish executable in a memorable location.

### 6. Configure Paths in `server.py`

Open the `server.py` file and update the following path to point to the location of your Stockfish executable:

```
# Make sure this path points to your actual Stockfish executable
STOCKFISH_PATH = "/path/to/your/stockfish/executable"
```

The server assumes your pre-trained models (`chess_imitator_final.pth` and `humanity_model_resnet_v5.pth`) are located in a `/models` sub-directory.

### 7. (Optional) Download Game Data for Training

The pre-trained models are already included. However, if you wish to retrain the models, you will need a large dataset of chess games.

1. Go to the Lichess Open Database: [**https://database.lichess.org/**](https://database.lichess.org/)
2. Download a `.pgn` database of games. For this project, high-rated games (e.g., Lichess Rated 2200+) are recommended.
3. Create a `data` folder in the project's root directory and place the downloaded `.pgn` files inside it.

### 8. (Optional) Training the Models

If you have downloaded a new dataset, you can train the models from scratch using the provided scripts. Ensure your `.pgn` files are in the `/data` directory.

```
# To train the stylistic personality model
python train_stylistic_model.py

# To train the humanity model
python train_humanity_model.py
```

Trained model files (`.pth`) will be saved in the `/models` directory.

## Running the Server

Once all the dependencies are installed and the paths are configured, you can start the Flask server.

1. Make sure your virtual environment is activated.
2. Run the following command in your terminal:

```
python server.py
```

The server will start, and you should see output indicating that it is running on `http://1227.0.0.1:5000`. You can now open the `index.html` frontend in your browser, and it will be able to communicate with your local AI engine.

## API Endpoint

The server exposes a single API endpoint to get the engine's move.

* **URL:** `/api/get-move`
* **Method:** `POST`
* **Body (JSON):**
  ```
  {
    "fen": "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2"
  }
  ```

* **Success Response (JSON):**
  ```
  {
    "best_move": "g1f3",
    "analysis": {
      "humanity": 0.039,
      "style_score": 0.452,
      "tactic_score": 0.069
    }
  }
  ```
