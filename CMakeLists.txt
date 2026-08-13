cmake_minimum_required(VERSION 4.1)
project(lumia)

set(CMAKE_CXX_STANDARD 20)

add_executable(lumia
        src/main.cpp
        src/tokenizer/bpe_tokenizer.cpp
        include/tokenier/corpus.h
        src/tokenizer/corpus.cpp
)

include_directories(
        include
)