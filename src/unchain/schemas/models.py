from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class ModelCapabilities:
    """Model capabilities configuration."""
    
    provider: str
    max_context_window_tokens: int
    supports_tools: bool = True
    supports_response_format: bool = False
    supports_previous_response_id: bool = False
    supports_reasoning: bool = False
    input_modalities: List[str] = field(default_factory=lambda: ["text"])
    input_source_types: Dict[str, List[str]] = field(default_factory=dict)
    allowed_payload_keys: List[str] = field(default_factory=list)
    provider_model: Optional[str] = None
    model_type: Optional[str] = None
    default_embedding_dimensions: Optional[int] = None
    supports_dimensions: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "provider": self.provider,
            "max_context_window_tokens": self.max_context_window_tokens,
            "supports_tools": self.supports_tools,
            "supports_response_format": self.supports_response_format,
            "supports_previous_response_id": self.supports_previous_response_id,
            "supports_reasoning": self.supports_reasoning,
            "input_modalities": self.input_modalities,
            "input_source_types": self.input_source_types,
            "allowed_payload_keys": self.allowed_payload_keys,
        }
        
        if self.provider_model is not None:
            result["provider_model"] = self.provider_model
        if self.model_type is not None:
            result["model_type"] = self.model_type
        if self.default_embedding_dimensions is not None:
            result["default_embedding_dimensions"] = self.default_embedding_dimensions
        if self.supports_dimensions is not None:
            result["supports_dimensions"] = self.supports_dimensions
            
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCapabilities":
        """Create from dictionary representation."""
        return cls(**data)


@dataclass
class ModelDefaultPayload:
    """Default payload configuration for a model."""
    
    payload: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.payload.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelDefaultPayload":
        """Create from dictionary representation."""
        return cls(payload=data)


@dataclass
class ModelConfiguration:
    """Complete model configuration including capabilities and defaults."""
    
    name: str
    capabilities: ModelCapabilities
    default_payload: ModelDefaultPayload
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "capabilities": self.capabilities.to_dict(),
            "default_payload": self.default_payload.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, name: str, capabilities_data: Dict[str, Any], payload_data: Dict[str, Any]) -> "ModelConfiguration":
        """Create from dictionary representations."""
        return cls(
            name=name,
            capabilities=ModelCapabilities.from_dict(capabilities_data),
            default_payload=ModelDefaultPayload.from_dict(payload_data)
        )


# Pre-defined model configurations
CLAUDE_HAIKU_45 = ModelConfiguration(
    name="claude-haiku-4.5",
    capabilities=ModelCapabilities(
        provider="anthropic",
        provider_model="claude-haiku-4-5-20251001",
        max_context_window_tokens=200000,
        supports_tools=True,
        supports_response_format=False,
        supports_previous_response_id=False,
        supports_reasoning=False,
        input_modalities=["text", "image"],
        input_source_types={
            "image": ["url", "base64"]
        },
        allowed_payload_keys=["max_tokens", "temperature", "top_k", "top_p"]
    ),
    default_payload=ModelDefaultPayload(
        payload={
            "max_tokens": 32000,
            "temperature": 0.7,
        }
    )
)

GPT_41_MINI = ModelConfiguration(
    name="gpt-4.1-mini",
    capabilities=ModelCapabilities(
        provider="openai",
        provider_model="gpt-4.1-mini",
        max_context_window_tokens=1047576,
        supports_tools=True,
        supports_response_format=True,
        supports_previous_response_id=True,
        supports_reasoning=False,
        input_modalities=["text", "image", "pdf"],
        input_source_types={
            "image": ["url", "base64"],
            "pdf": ["url", "base64"]
        },
        allowed_payload_keys=["instructions", "temperature", "top_p", "max_output_tokens", "truncation", "tool_choice"]
    ),
    default_payload=ModelDefaultPayload(
        payload={
            "instructions": "",
            "temperature": 0.7,
            "top_p": 1,
            "max_output_tokens": 32768,
            "truncation": "auto",
        }
    )
)

GPT_4O = ModelConfiguration(
    name="gpt-4o",
    capabilities=ModelCapabilities(
        provider="openai",
        provider_model="gpt-4o",
        max_context_window_tokens=128000,
        supports_tools=True,
        supports_response_format=True,
        supports_previous_response_id=False,
        supports_reasoning=False,
        input_modalities=["text", "image"],
        input_source_types={
            "image": ["url", "base64"]
        },
        allowed_payload_keys=["max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty", "response_format"]
    ),
    default_payload=ModelDefaultPayload(
        payload={
            "max_tokens": 16384,
            "temperature": 0.7,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
    )
)

CLAUDE_SONNET_35 = ModelConfiguration(
    name="claude-sonnet-3.5",
    capabilities=ModelCapabilities(
        provider="anthropic",
        provider_model="claude-3-5-sonnet-20241022",
        max_context_window_tokens=200000,
        supports_tools=True,
        supports_response_format=False,
        supports_previous_response_id=False,
        supports_reasoning=True,
        input_modalities=["text", "image"],
        input_source_types={
            "image": ["url", "base64"]
        },
        allowed_payload_keys=["max_tokens", "temperature", "top_k", "top_p"]
    ),
    default_payload=ModelDefaultPayload(
        payload={
            "max_tokens": 32000,
            "temperature": 0.7,
            "top_p": 1
        }
    )
)

GEMINI_PRO_25 = ModelConfiguration(
    name="gemini-2.5-pro",
    capabilities=ModelCapabilities(
        provider="gemini",
        provider_model="gemini-2.5-pro",
        max_context_window_tokens=1048576,
        supports_tools=True,
        supports_response_format=True,
        supports_previous_response_id=False,
        supports_reasoning=True,
        input_modalities=["text", "image", "pdf"],
        input_source_types={"image": ["url", "base64"], "pdf": ["url", "base64"]},
        allowed_payload_keys=[
            "max_output_tokens",
            "temperature",
            "top_k",
            "top_p",
            "thinking_config",
        ],
    ),
    default_payload=ModelDefaultPayload(
        payload={"max_output_tokens": 65536, "temperature": 0.7, "top_p": 1}
    ),
)
# Deprecated import alias; it resolves to the supported Gemini configuration.
GEMINI_PRO_15 = GEMINI_PRO_25

__all__ = [
    "ModelCapabilities",
    "ModelDefaultPayload",
    "ModelConfiguration",
    "CLAUDE_HAIKU_45",
    "GPT_41_MINI",
    "GPT_4O",
    "CLAUDE_SONNET_35",
    "GEMINI_PRO_15",
    "GEMINI_PRO_25",
]
