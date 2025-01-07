enable_testing()

# Add doctest include directory
include_directories(${doctest_SOURCE_DIR}/doctest)

# Configure test executables
add_executable(tensor_tests tensor_tests.cpp)
target_link_libraries(tensor_tests PRIVATE dl doctest::doctest)
target_include_directories(tensor_tests PRIVATE ${doctest_SOURCE_DIR}/doctest)

add_executable(ops_tests ops_tests.cpp)
target_link_libraries(ops_tests PRIVATE dl doctest::doctest)
target_include_directories(ops_tests PRIVATE ${doctest_SOURCE_DIR}/doctest)

add_executable(matrix_ops_tests matrix_ops_tests.cpp)
target_link_libraries(matrix_ops_tests PRIVATE dl doctest::doctest)
target_include_directories(matrix_ops_tests PRIVATE ${doctest_SOURCE_DIR}/doctest)

add_executable(linear_tests linear_tests.cpp)
target_link_libraries(linear_tests PRIVATE dl doctest::doctest)
target_include_directories(linear_tests PRIVATE ${doctest_SOURCE_DIR}/doctest)

# Register tests
add_test(NAME tensor_tests COMMAND tensor_tests)
add_test(NAME ops_tests COMMAND ops_tests)
add_test(NAME matrix_ops_tests COMMAND matrix_ops_tests)
add_test(NAME linear_tests COMMAND linear_tests)
