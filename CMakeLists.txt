cmake_minimum_required(VERSION 3.16.0)
project(svgpm VERSION 0.1.0)

include(CTest)
enable_testing()

add_subdirectory(svgpm-libs)
add_subdirectory(app)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

