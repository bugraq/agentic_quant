"""LLM soyutlaması — sağlayıcı arayüzü (dummy / openrouter / vllm)."""
from llm.providers import DummyProvider, HypothesisProvider, make_critic, make_provider

__all__ = ["HypothesisProvider", "DummyProvider", "make_provider", "make_critic"]
