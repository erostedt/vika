file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "*.cu")

add_executable(run_tests ${TEST_SOURCES})
target_include_directories(run_tests PRIVATE ${PROJECT_SOURCE_DIR}/third_party ..)

enable_testing()
add_test(NAME run_tests COMMAND run_tests)
