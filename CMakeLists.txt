cmake_minimum_required(VERSION 4.1)
project(lumia)

set(CMAKE_CXX_STANDARD 20)

add_library(lumia
        src/main.cpp
        src/tokenizer/bpe_tokenizer.cpp
        src/tokenizer/corpus.cpp
        tests/bpe_tokenizer.cpp
        src/logger.cpp
        src/attention/self-attention.cpp
        src/attention/multi-head-attention.cpp
        src/layers/layer_norm.cpp
        src/layers/linear.cpp
        include/lumia/layers/linear.h
)

include_directories(
        include
)

target_include_directories( lumia
        PUBLIC
        external/spdlog/include
        external/eigen
)

add_executable(lumia-exe
        src/main.cpp
)

target_link_libraries(lumia-exe
        PRIVATE
        lumia
)

add_executable(lumia-test
        tests/bpe_tokenizer.cpp
)

target_link_libraries(lumia-test
        PRIVATE
        lumia
)



