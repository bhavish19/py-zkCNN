cmake_minimum_required(VERSION 3.10)
project(zkcnn_backend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GMP REQUIRED gmp)
pkg_check_modules(OPENSSL REQUIRED openssl)

# Include directories
include_directories(${GMP_INCLUDE_DIRS})
include_directories(${OPENSSL_INCLUDE_DIRS})

# Create shared library
add_library(zkcnn_backend SHARED
    zkcnn_backend.cpp
)

# Link libraries
target_link_libraries(zkcnn_backend
    ${GMP_LIBRARIES}
    ${OPENSSL_LIBRARIES}
)

# Set output name
set_target_properties(zkcnn_backend PROPERTIES
    OUTPUT_NAME "zkcnn"
    PREFIX "lib"
)

# Install rules
install(TARGETS zkcnn_backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Copy library to project root for easy access
# Note: This is handled by the build script instead



