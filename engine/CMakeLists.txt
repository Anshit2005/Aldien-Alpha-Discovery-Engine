cmake_minimum_required(VERSION 3.16)

project(alpha_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use CMake's FindPython (the old FindPythonLibs is deprecated). Pass
# -DPython_EXECUTABLE=... to build for a specific interpreter.
set(PYBIND11_FINDPYTHON ON)
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
    src/Engine.cpp
)


# ------------------------------------------------------------
# MinGW (MSYS2 UCRT64) on Windows
#
# The official Python for Windows does not ship the GCC runtime DLLs
# (libstdc++, libgcc, winpthread), so link them into every target;
# otherwise `import alpha_engine_cpp` fails with "DLL load failed".
# ------------------------------------------------------------

if(MINGW)
    foreach(target alpha_engine alpha_engine_tests alpha_engine_cpp)
        target_link_options(${target} PRIVATE -static-libgcc -static-libstdc++ -static)
    endforeach()
endif()