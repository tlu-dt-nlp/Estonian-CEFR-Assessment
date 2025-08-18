import json
from dataclasses import dataclass, asdict, field
from typing import Optional, List

from pydantic import BaseModel, Field

from gec_worker import worker_config


class Request(BaseModel):
    """
    A dataclass to store requests
    """
    text: str = Field(..., max_length=worker_config.max_input_length)
    language: str = "et"


@dataclass
class Span:
    start: int
    end: int
    value: str


@dataclass
class Replacement:
    value: Optional[str] = None


@dataclass
class Correction:
    span: Span
    replacements: List[Replacement]


@dataclass
class Response:
    """
    A dataclass that can be used to store responses and transfer them over the message queue if needed.
    """
    corrections: [List[Correction]] = field(default_factory=list)
    corrected_text: Optional[str] = None
    original_text: Optional[str] = None
    status_code: int = 200
    status: str = 'OK'

    def encode(self) -> bytes:
        return json.dumps(asdict(self)).encode()
