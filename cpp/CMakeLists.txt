cmake_minimum_required(VERSION 3.18)
project(learned_bloom_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static link MinGW runtime to avoid DLL dependencies
if(MINGW)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(learned_bloom_cpp src/pybind_module.cpp src/bloom_filter.cpp src/fused_trigram_scorer.cpp)
target_include_directories(learned_bloom_cpp PRIVATE include)
