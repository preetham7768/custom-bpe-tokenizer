import pytest
from bpe.tokenizer import BPETokenizer

def test_roundtrip_basic():
    text = "Hello world! This is a test for Byte Pair Encoding."
    tok = BPETokenizer()
    tok.train(text, vocab_size=265)
    
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    
    assert decoded == text
    assert len(encoded) < len(text.encode("utf-8"))

def test_special_tokens():
    text = "User message<|endoftext|>Assistant response"
    tok = BPETokenizer()
    tok.train(text, vocab_size=260)
    tok.register_special_tokens({"<|endoftext|>": 1000})

    encoded = tok.encode(text, allowed_special="all")
    assert 1000 in encoded
    assert tok.decode(encoded) == text

def test_multilingual_utf8():
    text = "नमस्ते दुनिया, Hello world, こんにちは世界 🚀"
    tok = BPETokenizer()
    tok.train(text, vocab_size=270)
    
    encoded = tok.encode(text)
    assert tok.decode(encoded) == text
