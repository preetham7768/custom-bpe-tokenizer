# Custom Byte-Pair Encoding (BPE) Tokenizer Engine

A lightweight, from-scratch implementation of the Byte Pair Encoding (BPE) subword tokenization algorithm used in modern GPT-style Large Language Models (GPT-2, GPT-4, Llama). 

Built without high-level tokenization frameworks to demonstrate core NLP foundations: raw UTF-8 byte manipulation, frequency-priority merges, regex pre-tokenization, and special token handling.

## Features
- **Zero-Dependency Core:** Implemented directly over Python's `zip`, dictionaries, and standard regex.
- **UTF-8 Byte Baseline:** Starts with 256 byte tokens to guarantee no out-of-vocabulary (OOV) errors.
- **GPT-4 Style Pre-Tokenization:** Regex-driven structural chunking prevents undesirable merges across whitespace, punctuation, and numeric boundaries.
- **Special Token Registry:** Full support for registering and parsing delimiter tokens (e.g., `<|endoftext|>`, `<|im_start|>`).
- **Lossless Reconstruction:** Guarantees string roundtrip consistency: `decode(encode(text)) == text`.

## Project Structure
```text
custom-bpe-tokenizer/
├── bpe/
│   ├── base.py         # Statistical counters and low-level merge logic
│   └── tokenizer.py    # Main BPETokenizer class (train, encode, decode, save/load)
├── tests/              # Pytest verification suites
├── requirements.txt
└── README.md
