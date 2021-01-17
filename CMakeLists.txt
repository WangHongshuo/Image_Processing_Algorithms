cmake_minimum_required(VERSION 3.0.0)
project(image_processing_algorithms VERSION 0.1.0)

set(OpenCV_DIR "D:/opencv")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_executable(image_canny "${PROJECT_SOURCE_DIR}/image_canny/image_canny.cpp")
target_link_libraries(image_canny ${OpenCV_LIBS})

add_executable(image_dft_idft "${PROJECT_SOURCE_DIR}/image_dft_idft/image_dft_idft.cpp")
target_link_libraries(image_dft_idft ${OpenCV_LIBS})

add_executable(image_resize "${PROJECT_SOURCE_DIR}/image_resize/image_resize.cpp")
target_link_libraries(image_resize ${OpenCV_LIBS})

add_executable(image_fft_ifft "${PROJECT_SOURCE_DIR}/image_fft_ifft/image_fft_ifft.cpp")
target_link_libraries(image_fft_ifft ${OpenCV_LIBS})

add_executable(image_hough "${PROJECT_SOURCE_DIR}/image_hough/image_hough.cpp")
target_link_libraries(image_hough ${OpenCV_LIBS})

add_executable(image_radon "${PROJECT_SOURCE_DIR}/image_radon/image_radon.cpp")
target_link_libraries(image_radon ${OpenCV_LIBS})

add_executable(image_white_balance "${PROJECT_SOURCE_DIR}/image_white_balance/image_white_balance.cpp")
target_link_libraries(image_white_balance ${OpenCV_LIBS})