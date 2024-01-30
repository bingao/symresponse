# Required version of CMake
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Project name
project(SymResponse VERSION 1.0.0 LANGUAGES CXX)

# Options for making the library
option(BUILD_SHARED_LIBS "Build shared library." OFF)
option(BUILD_TESTING "Build tests." ON)

# From https://gitlab.com/CLIUtils/modern-cmake.git
#
# FetchContent added in CMake 3.11, downloads during the configure step
# FetchContent_MakeAvailable was added in CMake 3.14; simpler usage
include(FetchContent)

# Find SymEngine, SymEngine_DIR can be set by users in advance
set(CMAKE_PREFIX_PATH ${SymEngine_DIR} ${CMAKE_PREFIX_PATH})
find_package(SymEngine REQUIRED CONFIG
             PATH_SUFFIXES lib/cmake/symengine cmake/symengine CMake/)
message("SymEngine_DIR: " ${SymEngine_DIR})
message("SymEngine version: " ${SymEngine_VERSION})

# Find Tinned, Tinned_DIR can be set by users in advance
set(CMAKE_PREFIX_PATH ${Tinned_DIR} ${CMAKE_PREFIX_PATH})
find_package(Tinned REQUIRED CONFIG
             PATH_SUFFIXES lib/cmake/tinned)
message("Tinned_DIR: " ${Tinned_DIR})
message("Tinned version: " ${Tinned_VERSION})

# Use the same build mode and C++ flags of SymEngine
set(CMAKE_BUILD_TYPE ${SYMENGINE_BUILD_TYPE})
set(CMAKE_CXX_FLAGS_RELEASE ${SYMENGINE_CXX_FLAGS_RELEASE})
set(CMAKE_CXX_FLAGS_DEBUG ${SYMENGINE_CXX_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYMENGINE_CXX_FLAGS}")

# Settings from host programs. Host programs can include this file by setting
# LIB_SYMRESPONSE_PATH
if("${LIB_SYMRESPONSE_PATH}" STREQUAL "")
    set(LIB_SYMRESPONSE_PATH ${PROJECT_SOURCE_DIR})
endif()
if(NOT "${PARENT_DEFINITIONS}" STREQUAL "")
    foreach(_definition ${PARENT_DEFINITIONS})
        add_definitions(${_definition})
    endforeach()
endif()
if(NOT "${PARENT_INCLUDE_DIR}" STREQUAL "")
    include_directories(${PARENT_INCLUDE_DIR})
endif()
#if(NOT "${PARENT_MODULE_DIR}" STREQUAL "")
#    set(CMAKE_Fortran_MODULE_DIRECTORY ${PARENT_MODULE_DIR})
#endif()

## Uninstalls target, modified from (accessing date 2015-09-27)
## http://cmake.org/Wiki/CMake_FAQ#Can_I_do_.22make_uninstall.22_with_CMake.3F
#configure_file("${LIB_SYMRESPONSE_PATH}/cmake/cmake_uninstall.cmake.in"
#               "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
#               @ONLY)
#add_custom_target(uninstall
#                  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

# Summary
message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message("CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message("CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message("LIB_SYMRESPONSE_PATH: ${LIB_SYMRESPONSE_PATH}")
message("SYMENGINE_INCLUDE_DIRS: ${SYMENGINE_INCLUDE_DIRS}")
message("SYMENGINE_LIBRARIES: ${SYMENGINE_LIBRARIES}")
#message("TINNED_INCLUDE_DIRS: ${TINNED_INCLUDE_DIRS}")
#message("TINNED_LIBRARIES: ${TINNED_LIBRARIES}")
message("BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message("BUILD_TESTING: ${BUILD_TESTING}")
