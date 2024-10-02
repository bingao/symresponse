# Adds Catch2::Catch2, from https://gitlab.com/CLIUtils/modern-cmake.git
FetchContent_Declare(
    catch
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v2.13.6)
FetchContent_MakeAvailable(catch)

add_executable(test_dao test_dao.cpp)
target_link_libraries(test_dao
                      PRIVATE symresponse tinned ${SYMENGINE_LIBRARIES} Catch2::Catch2)
add_test(NAME test_dao COMMAND test_dao)

add_executable(test_two_level test_two_level.cpp)
target_link_libraries(test_two_level
                      PRIVATE symresponse tinned ${SYMENGINE_LIBRARIES} Catch2::Catch2)
add_test(NAME test_two_level COMMAND test_two_level)

add_executable(test_cc test_cc.cpp)
target_link_libraries(test_cc
                      PRIVATE symresponse tinned ${SYMENGINE_LIBRARIES} Catch2::Catch2)
add_test(NAME test_cc COMMAND test_cc)
