cmake_minimum_required(VERSION 3.16)

project(alpha_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

include_directories(include)


# ------------------------------------------------------------
# Main executable
# ------------------------------------------------------------

add_executable(
    alpha_engine

    src/main.cpp
    src/AST.cpp
    src/Evaluator.cpp
    src/CSVLoader.cpp
    src/Fitness.cpp
)


# ------------------------------------------------------------
# C++ tests
# ------------------------------------------------------------

enable_testing()

add_executable(
    alpha_engine_tests

    tests/test_evaluator.cpp
    src/AST.cpp
    src/Evaluator.cpp
)

add_test(
    NAME evaluator_tests
    COMMAND alpha_engine_tests
)


# ------------------------------------------------------------
# Python module
# ------------------------------------------------------------

pybind11_add_module(
    alpha_engine_cpp

    src/bindings.cpp
    src/AST.cpp
    src/Evaluator.cpp
    src/Fitness.cpp
)