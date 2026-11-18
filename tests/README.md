# Tests

The root CMake build includes three C++ tests, with assertions enabled in Release, and registers them with CTest. Run them with `ctest --test-dir "$BUILD_DIR"`.

Set `QLEAF_BENCH` and `QLEAF_BUILDER` to the executables in an external build directory before running the Python tests. Those tests create fixtures and execute search/build code.
