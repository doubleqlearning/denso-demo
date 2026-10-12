from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

class Role(str, Enum):
    """Message role options"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]

class Message(BaseModel):
    role: ROLE_TYPE = Field(...)
    content: Optional[str] = Field(default=None)
    name: Optional[str] = Field(default=None)

    def __add__(self, other) -> List["Message"]:
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")

    def __radd__(self, other) -> List["Message"]:
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'")

    def to_dict(self) -> Dict[str, Any]:
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.name is not None:
            message["name"] = self.name
        return message

    @classmethod
    def user_message(cls, content: str) -> "Message":
        return cls(role=Role.USER, content=content)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(cls, content: Optional[str] = None) -> "Message":
        return cls(role=Role.ASSISTANT, content=content)