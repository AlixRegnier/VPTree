cmake_minimum_required(VERSION 3.18)

project(vptree LANGUAGES CXX)

option(VPTREE_BUILD_PYTHON
    "Build Python bindings"
    ${PROJECT_IS_TOP_LEVEL}
)

# Header-only C++ library.
#
# This requires C++17 or newer. A consumer using C++20, C++23, etc.
# can continue using its higher C++ standard.
add_library(vptree INTERFACE)

target_include_directories(vptree
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(vptree INTERFACE cxx_std_17)

# Python bindings
if(VPTREE_BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(vptree_python
        python/bindings.cpp
    )

    set_target_properties(vptree_python PROPERTIES
        OUTPUT_NAME vptree
    )

    target_link_libraries(vptree_python
        PRIVATE vptree
    )

    install(TARGETS vptree_python
        LIBRARY DESTINATION .
    )
endif()