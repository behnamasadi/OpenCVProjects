cmake_minimum_required(VERSION 3.25)

project(OpenCVProjects
    VERSION 0.1.0
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Language and build defaults
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(ProcessorCount)
ProcessorCount(N)
if(N GREATER 0 AND NOT DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
    set(CMAKE_BUILD_PARALLEL_LEVEL ${N})
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)
find_package(OpenCV  REQUIRED)
find_package(Eigen3  REQUIRED NO_MODULE)
find_package(Ceres   QUIET)
find_package(glog    QUIET)
find_package(gflags  QUIET)

include(FetchContent)
FetchContent_Declare(rerun_sdk
    URL https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip
)
FetchContent_MakeAvailable(rerun_sdk)

message(STATUS "OpenCV ${OpenCV_VERSION}  (${OpenCV_DIR})")
message(STATUS "Eigen3 ${EIGEN3_VERSION_STRING}")
if(Ceres_FOUND)
    message(STATUS "Ceres  ${Ceres_VERSION}")
endif()

# ---------------------------------------------------------------------------
# Shared warning policy (target-scoped, no global CXX_FLAGS)
# ---------------------------------------------------------------------------
add_library(project_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(project_warnings INTERFACE
        -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(project_warnings INTERFACE /W4)
endif()

# ---------------------------------------------------------------------------
# Examples — one executable per source file in src/
# ---------------------------------------------------------------------------
# Targets that additionally need rerun_sdk
set(RERUN_TARGETS
    triangulation
    virtual_stereo_vision_cameras
    camera_projection_matrix
    P3P
)

file(GLOB EXAMPLES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# sandbox.cpp is a personal scratch file; gitignored and handled separately
list(FILTER EXAMPLES EXCLUDE REGEX "/sandbox\\.cpp$")

foreach(src IN LISTS EXAMPLES)
    get_filename_component(name "${src}" NAME_WE)
    add_executable(${name} "${src}")
    target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(${name} PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
        Threads::Threads
        project_warnings
    )
    if(name IN_LIST RERUN_TARGETS)
        target_link_libraries(${name} PRIVATE rerun_sdk)
    endif()
endforeach()

# Personal scratch target — only built if the file is present (it's gitignored)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/sandbox.cpp")
    add_executable(sandbox src/sandbox.cpp)
    target_include_directories(sandbox PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(sandbox PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
        Threads::Threads
        project_warnings
    )
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
# Register tests under tests/ here once they exist, e.g.:
#   add_subdirectory(tests)
