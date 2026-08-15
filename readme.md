# Lumia

**A lightweight C++ framework for building Large Language Models from scratch.**

Lumia is an experimental framework focused on understanding and implementing the core building blocks of LLMs in C++.

## Current Status

Currently working on:

* Error Handling

## Done

Done as of now:
* add special chars as vocab , like white space tap etc, and if not mapped to any vocal , a special token id.
* Logging using spdlog
* BPE Tokenizer
* Vocabulary
* Basic project structure

## Planned

* Tensor operations
* Neural network layers
* Attention
* Transformer architecture
* Training
* Inference

## Goal

Build a modular framework where different LLM architectures can be created by combining reusable components.

```text
Tokenizer
    ↓
Tensor
    ↓
Neural Network
    ↓
Attention
    ↓
Transformer
    ↓
LLM
```
