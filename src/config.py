from dataclasses import dataclass


@dataclass
class Config:
    model_name: str = 'openai/qwen3:0.6b'
