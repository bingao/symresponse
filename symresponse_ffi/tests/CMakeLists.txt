cmake_minimum_required(VERSION 3.20)
project(symresponse_c_tests C)

# User options
set(WORKSPACE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH "Path to symresponse workspace root")
set(SYMRESPONSE_CRATE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "Path to symresponse_ffi crate")
set(TINNED_CRATE_DIR "" CACHE PATH "Path to local tinned_ffi crate (repo clone)")
set(CARGO_PROFILE "release-ffi" CACHE STRING "debug, release, or release-ffi")
set(SYMRESPONSE_FEATURES "c-headers" CACHE STRING "Cargo features for symresponse_ffi")
set(TINNED_FEATURES "c-headers" CACHE STRING "Cargo features for tinned_ffi")

# Target dir (Cargo's output)
if(DEFINED ENV{CARGO_TARGET_DIR})
  set(TARGET_DIR "$ENV{CARGO_TARGET_DIR}")
else()
  set(TARGET_DIR "${WORKSPACE_DIR}/target")
endif()

# Profile handling
if (CARGO_PROFILE STREQUAL "release")
  set(CARGO_BUILD_ARGS --release)
  set(PROFILE_DIR "release")
elseif (CARGO_PROFILE STREQUAL "debug")
  set(CARGO_BUILD_ARGS "")
  set(PROFILE_DIR "debug")
elseif (CARGO_PROFILE STREQUAL "release-ffi")
  set(CARGO_BUILD_ARGS --profile release-ffi)
  set(PROFILE_DIR "release-ffi")
else()
  message(FATAL_ERROR "Unknown CARGO_PROFILE='${CARGO_PROFILE}'. Use debug, release, or release-ffi.")
endif()

# Build SymResponse FFI library
add_custom_target(rust_lib_symresponse ALL
  COMMAND ${CMAKE_COMMAND} -E env
          RUSTFLAGS=$ENV{RUSTFLAGS}
          cargo build
          --manifest-path "${SYMRESPONSE_CRATE_DIR}/Cargo.toml"
          --features "${SYMRESPONSE_FEATURES}"
          ${CARGO_BUILD_ARGS}
  WORKING_DIRECTORY "${SYMRESPONSE_CRATE_DIR}"
  COMMENT "Building symresponse_ffi (${CARGO_PROFILE})"
  VERBATIM
)

# Build Tinned FFI library
if(NOT TINNED_CRATE_DIR)
  message(FATAL_ERROR
    "TINNED_CRATE_DIR is empty. Point it to your local tinned repo's tinned_ffi crate, "
    "so we can build and import the cdylib for C tests.")
endif()

add_custom_target(rust_lib_tinned ALL
  COMMAND ${CMAKE_COMMAND} -E env
          CARGO_TARGET_DIR=${TARGET_DIR}
          RUSTFLAGS=$ENV{RUSTFLAGS}
          cargo build
          --manifest-path "${TINNED_CRATE_DIR}/Cargo.toml"
          --features "${TINNED_FEATURES}"
          ${CARGO_BUILD_ARGS}
  WORKING_DIRECTORY "${TINNED_CRATE_DIR}"
  COMMENT "Building tinned_ffi (${CARGO_PROFILE})"
  VERBATIM
)

# Import the built shared libs
if (WIN32)
  set(SYMRESP_RUNTIME "${TARGET_DIR}/${PROFILE_DIR}/symresponse_ffi.dll")
  set(SYMRESP_IMPLIB  "${TARGET_DIR}/${PROFILE_DIR}/symresponse_ffi.lib")
  add_library(symresponse_ffi SHARED IMPORTED GLOBAL)
  set_target_properties(symresponse_ffi PROPERTIES
    IMPORTED_LOCATION "${SYMRESP_RUNTIME}"
    IMPORTED_IMPLIB   "${SYMRESP_IMPLIB}"
  )
elseif(APPLE)
  set(SYMRESP_RUNTIME "${TARGET_DIR}/${PROFILE_DIR}/libsymresponse_ffi.dylib")
  add_library(symresponse_ffi SHARED IMPORTED GLOBAL)
  set_target_properties(symresponse_ffi PROPERTIES
    IMPORTED_LOCATION "${SYMRESP_RUNTIME}"
  )
else()
  set(SYMRESP_RUNTIME "${TARGET_DIR}/${PROFILE_DIR}/libsymresponse_ffi.so")
  add_library(symresponse_ffi SHARED IMPORTED GLOBAL)
  set_target_properties(symresponse_ffi PROPERTIES
    IMPORTED_LOCATION "${SYMRESP_RUNTIME}"
  )
endif()
add_dependencies(symresponse_ffi rust_lib_symresponse)

if (WIN32)
  set(TINNED_RUNTIME "${TARGET_DIR}/${PROFILE_DIR}/tinned_ffi.dll")
  set(TINNED_IMPLIB  "${TARGET_DIR}/${PROFILE_DIR}/tinned_ffi.lib")
  add_library(tinned_ffi SHARED IMPORTED GLOBAL)
  set_target_properties(tinned_ffi PROPERTIES
    IMPORTED_LOCATION "${TINNED_RUNTIME}"
    IMPORTED_IMPLIB   "${TINNED_IMPLIB}"
  )
elseif(APPLE)
  set(TINNED_RUNTIME "${TARGET_DIR}/${PROFILE_DIR}/libtinned_ffi.dylib")
  add_library(tinned_ffi SHARED IMPORTED GLOBAL)
  set_target_properties(tinned_ffi PROPERTIES
    IMPORTED_LOCATION "${TINNED_RUNTIME}"
  )
else()
  set(TINNED_RUNTIME "${TARGET_DIR}/${PROFILE_DIR}/libtinned_ffi.so")
  add_library(tinned_ffi SHARED IMPORTED GLOBAL)
  set_target_properties(tinned_ffi PROPERTIES
    IMPORTED_LOCATION "${TINNED_RUNTIME}"
  )
endif()
add_dependencies(tinned_ffi rust_lib_tinned)

# C tests
add_executable(test_api
  "${SYMRESPONSE_CRATE_DIR}/tests/test_api.c"
  "${SYMRESPONSE_CRATE_DIR}/tests/test_lagrangian_dao.c"
  "${SYMRESPONSE_CRATE_DIR}/tests/all_tests.def"
)

# Include both headers (must already exist)
target_include_directories(test_api PRIVATE
  "${TINNED_CRATE_DIR}/include"
  "${SYMRESPONSE_CRATE_DIR}/include"
  "${SYMRESPONSE_CRATE_DIR}/tests"
)

# Link both FFIs (order: symresponse first, then tinned)
target_link_libraries(test_api PRIVATE symresponse_ffi tinned_ffi)

# Runtime search path
if (APPLE)
  set_target_properties(test_api PROPERTIES
    BUILD_RPATH "@loader_path"
    INSTALL_RPATH "@loader_path"
  )
elseif(UNIX AND NOT APPLE)
  set_target_properties(test_api PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
  )
endif()

# Copy both runtimes next to the test exe
add_custom_command(TARGET test_api POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${SYMRESP_RUNTIME}" "$<TARGET_FILE_DIR:test_api>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${TINNED_RUNTIME}" "$<TARGET_FILE_DIR:test_api>"
  COMMENT "Copying Rust runtimes next to test_api"
)