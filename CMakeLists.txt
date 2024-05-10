# Required version of CMake
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Project name
project(SymResponse VERSION 0.1.0 LANGUAGES CXX)

# Options for making the library
option(BUILD_SHARED_LIBS "Build shared library." OFF)
option(BUILD_TESTING "Build tests." ON)
#option(BUILD_EXAMPLES "Build examples." ON)

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
find_package(Tinned REQUIRED CONFIG PATH_SUFFIXES lib/cmake/tinned)
message("Tinned_DIR: " ${Tinned_DIR})

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

# Include Tinned and SymEngine header file directories
set(SYMRESPONSE_INCLUDE_DIRS ${LIB_SYMRESPONSE_PATH}/include)
include_directories(
    ${SYMRESPONSE_INCLUDE_DIRS}
    ${TINNED_INCLUDE_DIR}
    ${SYMENGINE_INCLUDE_DIRS}
)

# SymResponse library source codes
add_subdirectory(src)

# Testing only in the main project, and not if it is included through
# add_subdirectory(). Learnt from Modern CMake at
# https://gitlab.com/CLIUtils/modern-cmake
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME) AND BUILD_TESTING)
    include(CTest)
    add_subdirectory(tests)
endif()

# Use `CMAKE_INSTALL_DATADIR`
include(GNUInstallDirs)

# Generate a config file, modified from
# https://cmake.org/cmake/help/v3.14/module/CMakePackageConfigHelpers.html
# https://cmake.org/cmake/help/latest/guide/tutorial/Adding%20Export%20Configuration.html
#
# find_package(SymResponse REQUIRED)
# target_link_libraries(... symresponse)
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}
    CACHE PATH "Location of include directory")
install(TARGETS symresponse
        EXPORT SymResponseTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${INCLUDE_INSTALL_DIR})
# Install include headers
install(FILES ${SYMRESPONSE_INCLUDE_DIRS}/SymResponse.hpp
        DESTINATION ${INCLUDE_INSTALL_DIR})
# Pattern matching option from https://stackoverflow.com/a/67883070
install(DIRECTORY ${SYMENGINE_INCLUDE_DIRS}/SymResponse
        DESTINATION ${INCLUDE_INSTALL_DIR}
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.swp" EXCLUDE)
# Install generated `SymResponseTargets.cmake`
set(SYMRESPONSE_INSTALL_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/symresponse)
#set(SYMRESPONSE_INSTALL_CMAKE_DIR ${CMAKE_INSTALL_DATADIR}/symresponse/cmake)
install(EXPORT SymResponseTargets
        FILE SymResponseTargets.cmake
        DESTINATION ${SYMRESPONSE_INSTALL_CMAKE_DIR})
# Helpers functions for creating config files
include(CMakePackageConfigHelpers)
# Generate the config file that includes the exports
configure_package_config_file(
    ${LIB_SYMRESPONSE_PATH}/SymResponseConfig.cmake.in
    ${PROJECT_BINARY_DIR}/SymResponseConfig.cmake
    INSTALL_DESTINATION ${SYMRESPONSE_INSTALL_CMAKE_DIR}
    PATH_VARS INCLUDE_INSTALL_DIR
    #NO_SET_AND_CHECK_MACRO
    #NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
# Create a version file
write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/SymResponseConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)
install(FILES
        ${PROJECT_BINARY_DIR}/SymResponseConfig.cmake
        ${PROJECT_BINARY_DIR}/SymResponseConfigVersion.cmake
        DESTINATION ${SYMRESPONSE_INSTALL_CMAKE_DIR})

# `export` lets one use SymResponse without installing it
#export(TARGETS symresponse FILE "${PROJECT_BINARY_DIR}/SymResponseTargets.cmake")
export(EXPORT SymResponseTargets FILE "${PROJECT_BINARY_DIR}/SymResponseTargets.cmake")

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
message("Tinned include directory: ${TINNED_INCLUDE_DIR}")
#message("Tinned library: ${tinned}")
message("BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message("BUILD_TESTING: ${BUILD_TESTING}")
#message("BUILD_EXAMPLES: ${BUILD_EXAMPLES}")
