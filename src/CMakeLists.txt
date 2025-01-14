add_library(dl INTERFACE)

target_include_directories(dl
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

# Set up installation
install(TARGETS dl
    EXPORT dlTargets
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION include
)
