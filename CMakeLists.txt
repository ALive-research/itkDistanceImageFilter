cmake_minimum_required (VERSION 3.16)

project(itkDistanceImageFilter)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

find_package(ITK REQUIRED)
find_package(TCLAP REQUIRED)

include(${ITK_USE_FILE})

set(ITK_USE_KWSTYLE OFF)

# Add the ITK CMake directory to the CMake module path so other projects can use ITK's facilities
list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})

if(NOT ITK_SOURCE_DIR)
  include(ITKModuleExternal)
else()
  itk_module_impl()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} CACHE PATH "Executable output path" FORCE)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE main.cxx)
target_include_directories(${PROJECT_NAME} PRIVATE ${TCLAP_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${ITK_LIBRARIES})
