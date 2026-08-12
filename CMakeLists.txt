cmake_minimum_required(VERSION 4.1)
project(lumia)

set(CMAKE_CXX_STANDARD 20)

add_executable(lumia src/main.cpp
        include/tokenier/tokenizer.h)
