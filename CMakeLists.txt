cmake_minimum_required (VERSION 3.16)

project(itkDistanceMap)

find_package(ITK REQUIRED)

include(${ITK_USE_FILE})

set(ITK_USE_KWSTYLE OFF)

# Add the ITK CMake directory to the CMake module path so other projects can use ITK's facilities
list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})

if(NOT ITK_SOURCE_DIR)
  include(ITKModuleExternal)
else()
  itk_module_impl()
endif()

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE main.cxx)
target_link_libraries(${PROJECT_NAME} PRIVATE ${ITK_LIBRARIES})
