add_executable(train_mlp train_mlp.cpp)
target_link_libraries(train_mlp PRIVATE dl)
target_include_directories(train_mlp PRIVATE ${CMAKE_SOURCE_DIR}/include)
