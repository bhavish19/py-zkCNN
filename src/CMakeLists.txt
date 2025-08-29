# Get all source files except main files
aux_source_directory(. conv_src)
list(FILTER conv_src EXCLUDE REGEX "main*")

# Add timer source file
list(APPEND conv_src "../3rd/hyrax-bls12-381/src/timer.cpp")

# Create library
add_library(cnn_lib ${conv_src})

# Find GMP
find_package(PkgConfig REQUIRED)
pkg_check_modules(GMP REQUIRED gmp)

# Set include directories
target_include_directories(cnn_lib PRIVATE 
    ${GMP_INCLUDE_DIRS}
    ../3rd/hyrax-bls12-381/src
    ../3rd/hyrax-bls12-381/3rd/mcl/include
)

# Link libraries - use the libraries built in the current build directory
target_link_libraries(cnn_lib 
    ${GMP_LIBRARIES}
    hyrax_lib
    mcl
    mclbn256
    mclbn384
    mclbn384_256
)

# Create executables
add_executable(demo_lenet_run main_demo_lenet.cpp)
target_link_libraries(demo_lenet_run cnn_lib)

add_executable(demo_vgg_run main_demo_vgg.cpp)
target_link_libraries(demo_vgg_run cnn_lib)

add_executable(demo_alexnet_run main_demo_alexnet.cpp)
target_link_libraries(demo_alexnet_run cnn_lib)
