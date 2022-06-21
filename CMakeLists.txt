cmake_minimum_required(VERSION 3.10)
SET(CMAKE_BUILD_TYPE "Debug")

# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -Wextra -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -Werror -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

project(zkCNN)
set(CMAKE_CXX_STANDARD 14)

link_directories(3rd/hyrax-bls12-381)

include_directories(src)
include_directories(3rd)
include_directories(3rd/hyrax-bls12-381/3rd/mcl/include)

add_subdirectory(src)
add_subdirectory(3rd/hyrax-bls12-381)