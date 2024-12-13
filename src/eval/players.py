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
from src.network import GPT, GPTConfig


def filter_hyperparameters(hyper_parameters: dict, config_class) -> dict:
    valid_params = {
        field.name for field in config_class.__dataclass_fields__.values()
    }

    filtered_params = {
        k: v for k, v in hyper_parameters.items() if k in valid_params
    }

    return filtered_params


def replace_key_prefix(
    state_dict: dict, old_prefix: str, new_prefix: str
) -> dict:
    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith(old_prefix):
            new_key = key.replace(
                old_prefix, new_prefix, 1
            )  # only replace the first occurrence
        else:
            new_key = key  # keep the key unchanged if the prefix doesn't match

        new_state_dict[new_key] = value

    return new_state_dict


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
        # print(f"Game state: {game_state}")
        return board.san(result.move)

    def get_config(self) -> dict:
        return {"skill_level": self._skill_level, "play_time": self._play_time}

    def close(self):
        self._engine.quit()


class ChessGPTPlayer(pl.LightningModule, Player):
    def __init__(
        self,
        checkpoint_name: str = "gpt2-stockfish.ckpt",
        checkpoint_path: str = "checkpoints/gpt2-stockfish",
        activation_name: Optional[str] = None,
        activation_coefficient: Optional[float] = None,
        meta_path: str = "data/stockfish/meta.pkl",
        load_meta: bool = True,
        # dtype: str = "float16",
        device: str = "cpu",
    ):
        super().__init__()
        self.checkpoint_name = checkpoint_name
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

        ckpt = torch.load(
            self.checkpoint_path / self.checkpoint_name,
            map_location=self.device_type,
        )
        filtered_hparams = filter_hyperparameters(
            ckpt["hyper_parameters"], GPTConfig
        )

        gptconf = GPTConfig(**filtered_hparams)
        state_dict = replace_key_prefix(ckpt["state_dict"], "model.", "")

        # print(state_dict)
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        self.model = GPT(gptconf)
        self.model.load_state_dict(state_dict)

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

    @staticmethod
    def clean_game_state(game_state: str) -> str:
        # Split the game state into individual lines
        lines = game_state.splitlines()

        # Skip the first two lines
        if len(lines) > 2:
            cleaned_lines = lines[2:]
        else:
            # Handle the case where there are fewer than two lines, if needed
            cleaned_lines = []

        # Join the remaining lines back into a single string
        cleaned_game_state = "\n".join(cleaned_lines)

        return cleaned_game_state

    def get_nanogpt_response(self, game_state: str, temperature: float) -> str:
        num_samples = 1
        top_k = 200
        max_new_tokens = 10

        game_state = self.clean_game_state(game_state)
        game_state = re.sub(r"(\d+\.) ", r"\1", game_state)
        game_state = ";" + game_state
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
        self,
        board: chess.Board,
        game_state: str,
        temperature: float,
        max_attempts: int = 20,
    ) -> str:
        for attempt in range(max_attempts):
            temperature = temperature + random.uniform(-0.01, 0.01) * attempt
            completion = self.get_nanogpt_response(game_state, temperature)
            move = self.get_move_from_response(completion)
            try:
                move_uci = board.parse_san(move)
                if move_uci in board.legal_moves:
                    return move
            except Exception:
                print(
                    f"Attempt {attempt + 1}: Invalid move '{move}' generated, trying again."
                )
                continue
        return ""

    def get_config(self) -> dict:
        # TODO: add more information about the model, state, etc
        return {"model": f"chessgpt_{self.checkpoint_name}"}
