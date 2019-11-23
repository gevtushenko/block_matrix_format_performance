project(common)

set(COMMON_SOURCES
        matrix_converters.h
        measurement_class.cpp
        measurement_class.h)

add_library(common ${COMMON_SOURCES})
target_include_directories(common PUBLIC .)
