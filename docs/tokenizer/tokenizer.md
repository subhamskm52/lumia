# Tokenizer
The tokenizer converts raw text into a sequence of token IDs that can be consumed by an LLM.

## Design
a generic `Tokenizer` interface so different tokenization algorithms can be implemented independently.

```text
   tokenizer (interface)
   ├──  bpe_tokenizer(impl)
   └─  others.
```
## Directory Structure

```text
include/lumia/tokenizer/
├── tokenizer.h
└── vocabulary.h

src/tokenizer/
├── tokenizer.cpp
└── vocabulary.cpp
```
