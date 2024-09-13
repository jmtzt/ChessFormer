import pickle
import platform
import random
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import pytorch_lightning as pl
import tiktoken
import torch
from torch import autocast

from src.config import ModelConfig
from src.model import GPTChessLightning


class Player:
    def get_move(
        self, board: chess.Board, game_state: str, temperature: float
    ) -> Optional[str]:
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError


class StockfishPlayer(Player):
    @staticmethod
    def get_stockfish_path() -> str:
        """
        Determines the operating system and returns the appropriate path for
        Stockfish.

        Returns:
            str: Path to the Stockfish executable based on the operating
                system.
        """
        if platform.system() == "Linux":
            return "/usr/games/stockfish"
        elif platform.system() == "Darwin":
            return "stockfish"
        elif platform.system() == "Windows":
            return r"C:\Users\user\Documents\Stockfish\stockfish-windows-x86-64-avx2.exe"
        else:
            raise OSError("Unsupported operating system")

    def __init__(self, skill_level: int, play_time: float):
        self._skill_level = skill_level
        self._play_time = play_time
        # If getting started, you need to run brew install stockfish
        stockfish_path = StockfishPlayer.get_stockfish_path()
        self._engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def get_move(
        self, board: chess.Board, game_state: str, temperature: float
    ) -> Optional[str]:
        if self._skill_level == -2:
            legal_moves = list(board.legal_moves)
            random_move = random.choice(legal_moves)
            return board.san(random_move)
        elif self._skill_level < 0:
            self._engine.configure({"Skill Level": 0})
            result = self._engine.play(
                board, chess.engine.Limit(time=1e-8, depth=1, nodes=1)
            )

        else:
            self._engine.configure({"Skill Level": self._skill_level})
            result = self._engine.play(
                board, chess.engine.Limit(time=self._play_time)
            )
        if result.move is None:
            return None
        print(f"Game state: {game_state}")
        return board.san(result.move)

    def get_config(self) -> dict:
        return {"skill_level": self._skill_level, "play_time": self._play_time}

    def close(self):
        self._engine.quit()


class ChessGPTPlayer(pl.LightningModule, Player):
    def __init__(
        self,
        model_name: str = "last-v1.ckpt",
        checkpoint_path: str = "checkpoints/trained",
        activation_name: Optional[str] = None,
        activation_coefficient: Optional[float] = None,
        meta_path: str = "data/lichess/meta.pkl",
        load_meta: bool = True,
        # dtype: str = "float16",
        device: str = "cpu",
    ):
        super().__init__()
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)
        self.activation_name = activation_name
        self.activation_coefficient = activation_coefficient
        self.meta_path = Path(meta_path)
        self.load_meta = load_meta
        # self.dtype = dtype
        self.device_type = (
            "cuda" if torch.cuda.is_available() and "cuda" in device else "cpu"
        )
        # self.autocast_dtype = {
        #     "float32": torch.float32,
        #     "bfloat16": torch.bfloat16,
        #     "float16": torch.float16,
        # }[dtype]

        self.model, self.config = self.load_model_from_checkpoint(
            self.checkpoint_path / self.model_name
        )
        self.model.to(self.device_type)
        self.model.eval()
        self.save_hyperparameters()

        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else autocast(
                device_type=self.device_type, dtype=self.autocast_dtype
            )
        )

        if self.load_meta:
            self.load_meta_encodings()
        else:
            self.setup_default_encodings()

    def load_meta_encodings(self):
        if self.meta_path.exists():
            print(f"Loading meta from {self.meta_path}...")
            with self.meta_path.open("rb") as f:
                meta = pickle.load(f)
            stoi, itos = meta["stoi"], meta["itos"]
            self.encode = lambda string: [stoi[c] for c in string]
            self.decode = lambda number: "".join([itos[i] for i in number])
        else:
            print(
                f"No meta file found at {meta_path}, using default GPT-2 encodings..."
            )
            self.setup_default_encodings()

    def setup_default_encodings(self):
        enc = tiktoken.get_encoding("gpt2")
        self.encode = lambda string: enc.encode(string, allowed_special={""})
        self.decode = lambda number: enc.decode(number)

    def load_model_from_checkpoint(self, ckpt_path: Path):
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device_type)
            config = checkpoint["hyper_parameters"]
            config = ModelConfig(**config)
            model = GPTChessLightning(config=config)

            state_dict = checkpoint["state_dict"]

            unwanted_prefix = "_orig_mod."
            for k, _ in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

            model.load_state_dict(state_dict)

            # if self.activation_name:
            #     # TODO: implement this logic for visualization purposes later...
            #     pass
            return model, config
        else:
            # TODO: implement this logic to load from pre-trained gpt2 models
            # possible values: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
            # model = GPT.from_pretrained(self.model_name)
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    def get_nanogpt_response(self, game_state: str, temperature: float) -> str:
        num_samples = 1
        top_k = 200
        max_new_tokens = 10

        print(f"game_state before: {game_state}")
        # ChessGPT model was trained only on pgn transcripts, so we need to
        # remove the stockfish evaluations from the game_state
        # e.g.: ["stockfish elo xxx"]\n["stockfish elo xxx"]\n\n
        game_state = game_state.split("\n\n")[1].strip()

        print(f"game_state after split: {game_state}")

        # We remove the space after the move number to match the training data
        # 1.e4 e5 2.Nf3, and not 1. e4 e5 2. Nf3
        game_state = re.sub(r"(\d+\.) ", r"\1", game_state)

        game_state = ";" + game_state

        print(f"game_state after: {game_state}")

        start_ids = self.encode(game_state)

        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[
            None, ...
        ]
        model_response = ""
        with torch.no_grad():
            with self.ctx:
                for _ in range(num_samples):
                    y = self.model.generate(
                        x, max_new_tokens, temperature=temperature, top_k=top_k
                    )

                    model_response = self.decode(y[0].tolist())

        print(f"model_response: {model_response}")
        # model_response includes the input string
        model_response = model_response[len(game_state) :]
        if ";" in model_response:
            model_response = model_response.split(";")[0]
        return model_response

    def get_move_from_response(self, response: str) -> str:
        # Extract and return the first move from the response
        moves = response.split()
        return moves[0] if moves else ""

    def get_move(
        self, board: chess.Board, game_state: str, temperature: float
    ) -> str:
        completion = self.get_nanogpt_response(game_state, temperature)
        return self.get_move_from_response(completion)

    def get_config(self) -> dict:
        # TODO: add more information about the model, state, etc
        return {"model_name": self.model_name}
