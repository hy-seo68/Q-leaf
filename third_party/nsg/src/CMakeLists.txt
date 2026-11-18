set(CMAKE_CXX_STANDARD 11)


set(COMMON_SOURCES
    index.cpp
)

set(SOURCES_NO_EARLY_TERM
    ${COMMON_SOURCES}
    index_nsg.cpp
)








add_library(${PROJECT_NAME} ${SOURCES_NO_EARLY_TERM})
add_library(${PROJECT_NAME}_s STATIC ${SOURCES_NO_EARLY_TERM})





#install()
