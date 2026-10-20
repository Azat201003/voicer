cmake_minimum_required(VERSION 3.10...4.4)

project(Voicer CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(voicer main.cpp network.cpp layer.cpp)

target_compile_options(voicer PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(voicer PRIVATE quadmath)
