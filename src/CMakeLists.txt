add_library(dl
    tensor.cpp
)

target_include_directories(dl
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# Set up installation
install(TARGETS dl
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION include
)
