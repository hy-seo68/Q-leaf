set(CMAKE_CXX_STANDARD 11)


find_library(TCMALLOC_LIB tcmalloc)
if(TCMALLOC_LIB)
    message(STATUS "Found TCMalloc: ${TCMALLOC_LIB}")
    set(TCMALLOC_LINK tcmalloc)
else()
    message(STATUS "TCMalloc not found - building without it")
    set(TCMALLOC_LINK "")
endif()

add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME} ${TCMALLOC_LINK})

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME})

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} ${TCMALLOC_LINK})

