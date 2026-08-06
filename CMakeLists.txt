cmake_minimum_required(VERSION 3.18)

project(vptree LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(vptree
    python/bindings.cpp
)

target_include_directories(vptree PRIVATE include)

target_compile_features(vptree PRIVATE cxx_std_17)

install(TARGETS vptree LIBRARY DESTINATION .)