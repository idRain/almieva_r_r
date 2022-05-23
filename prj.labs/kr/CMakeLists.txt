project(kr)
add_executable(kr kr.cpp kr.report.md.in.txt)
target_link_libraries(kr ${OpenCV_LIBS})

# prepare report md file
file(READ kr.cpp cpp_source)
configure_file(kr.report.md.in.txt ${CMAKE_CURRENT_BINARY_DIR}/kr.report.md @ONLY)
add_custom_command(TARGET ${PROJECT_NAME}
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/kr.report.md" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")