add_library(symresponse
            ${LIB_SYMRESPONSE_PATH}/src/LagrangianDAO.cpp)

target_include_directories(symresponse INTERFACE
    $<BUILD_INTERFACE:${SYMRESPONSE_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
