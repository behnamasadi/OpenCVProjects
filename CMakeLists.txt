cmake_minimum_required(VERSION 3.1)
set(CMAKE_BUILD_TYPE Debug)
option(USE_CLANG "build application with clang" OFF)



project(OpenCVProjects)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	message("using Clang")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	message("using GCC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	message("using Intel C++")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	message("using Visual Studio C++")

	# set warning level into errors
	if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
		string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	endif()
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")


include(FetchContent)
FetchContent_Declare(rerun_sdk URL https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)


message("\n########################################## glog ##########################################\n")

if(EXISTS "$ENV{HOME}/usr/lib/cmake/glog/")
	set(glog_DIR "$ENV{HOME}/usr/lib/cmake/glog/")
	find_package(glog)
	message("GLOG_FOUND: " ${glog_FOUND})
	message("GLOG_include_DIRS: " ${GLOG_include_DIRS})
	message("GLOG_LIBRARIES: " ${GLOG_LIBRARY})
else()
	message("glog not found")
endif()

# message("\n########################################## Ceres ##########################################\n")
# add_definitions(-DCERES_GFLAGS_NAMESPACE=${GFLAGS_NAMESPACE})

# if(EXISTS "$ENV{HOME}/usr/lib/cmake/Ceres")
# set(Ceres_DIR "$ENV{HOME}/usr/lib/cmake/Ceres")
# set(Ceres_DIR "$ENV{HOME}/usr/lib/cmake/Ceres")
# find_package(Ceres REQUIRED)
# message("CERES_FOUND: " ${CERES_FOUND})
# message("CERES_VERSION: " ${CERES_VERSION})
# message("CERES_LIBRARIES: " ${CERES_LIBRARIES})
# endif()

# find_package(Ceres QUIET)

# if(${CERES_FOUND})
# message("Ceres_VERSION: " ${Ceres_VERSION})
# message("CERES_include_DIRS: " ${CERES_include_DIRS})
# message("CERES_LIBRARIES: " ${CERES_LIBRARIES})

# # add_executable(bundle_adjuster_example1 src/bundle_adjuster_example1.cpp)
# # target_link_libraries(bundle_adjuster_example1  ${OpenCV_LIBS}  ${CERES_LIBRARIES})

# # add_executable(bundle_adjuster_example2 src/bundle_adjuster_example2.cpp src/bal_problem.cpp)
# # target_link_libraries(bundle_adjuster_example2  ${OpenCV_LIBS}  ${CERES_LIBRARIES} gflags)
# else()
# message("Ceres not found")
# endif()
message("\n########################################## Eigen ##########################################\n")

find_package(Eigen3)

message("EIGEN3_FOUND: " ${EIGEN3_FOUND})
message("EIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})
message("EIGEN3_VERSION: " ${EIGEN3_VERSION})
include_directories(${EIGEN3_INCLUDE_DIR})

message("\n########################################## gflag ##########################################\n")

if(EXISTS "$ENV{HOME}/usr/lib/cmake/gflags/")
	set(gflags_DIR "$ENV{HOME}/usr/lib/cmake/gflags/")
endif()

find_package(gflags QUIET)

if(${GFLAGS_FOUND})
	message("GFLAGS_FOUND: " ${GFLAGS_FOUND})
	message("GFLAGS_include_DIRS: " ${GFLAGS_include_DIRS})
	message("GFLAGS_LIBRARIES: " ${GFLAGS_LIBRARIES})
	message("GFLAGS_LIBRARYRARY_DIRS: " ${GFLAGS_LIBRARYRARY_DIRS})
	add_definitions(-DCERES_GFLAGS_NAMESPACE=${GFLAGS_NAMESPACE})
else()
	message("gflags not found")
endif()

message("\n########################################## OpenCV ##########################################\n")

if(EXISTS "$ENV{HOME}/usr/lib/cmake/opencv4")
	set(OpenCV_DIR "$ENV{HOME}/usr/lib/cmake/opencv4")
endif()

find_package(OpenCV REQUIRED)

if(${OpenCV_FOUND})
	message("OpenCV_DIR: " ${OpenCV_DIR})
	message("OpenCV_VERSION: " ${OpenCV_VERSION})
	message("OpenCV_include_DIRS: " ${OpenCV_include_DIRS})
	message("OpenCV_LIBS: " ${OpenCV_LIBS})
else()
	message("OpenCV not found")
endif()

add_executable(basic_operations src/basic_operations.cpp)
target_link_libraries(basic_operations ${OpenCV_LIBS})

add_executable(feature_detection_description src/feature_detection_description.cpp)
target_link_libraries(feature_detection_description ${OpenCV_LIBS})

add_executable(correspondences_matching src/correspondences_matching.cpp)
target_link_libraries(correspondences_matching ${OpenCV_LIBS})


add_executable(epipolar_geometry src/epipolar_geometry.cpp)
target_link_libraries(epipolar_geometry ${OpenCV_LIBS})


add_executable(essential_matrix_estimation src/essential_matrix_estimation.cpp)
target_link_libraries(essential_matrix_estimation ${OpenCV_LIBS})

add_executable(fundamental_matrix_estimation src/fundamental_matrix_estimation.cpp)
target_link_libraries(fundamental_matrix_estimation ${OpenCV_LIBS})


add_executable(coordinate_system_points src/coordinate_system_points.cpp)
target_link_libraries(coordinate_system_points ${OpenCV_LIBS})

add_executable(hough_transform src/hough_transform.cpp)
target_link_libraries(hough_transform ${OpenCV_LIBS})



add_executable(affine_transformation src/affine_transformation.cpp)
target_link_libraries(affine_transformation ${OpenCV_LIBS})

# add_executable(icp src/icp.cpp)
# target_link_libraries(icp ${OpenCV_LIBS})



add_executable(homogeneous_conversion src/homogeneous_conversion.cpp)
target_link_libraries(homogeneous_conversion ${OpenCV_LIBS})

add_executable(pca src/pca.cpp)
target_link_libraries(pca ${OpenCV_LIBS})

add_executable(morphological_transformation src/morphological_transformation.cpp)
target_link_libraries(morphological_transformation ${OpenCV_LIBS})

add_executable(image_thresholding src/image_thresholding.cpp)
target_link_libraries(image_thresholding ${OpenCV_LIBS})



add_executable(optical_flow src/optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS})

add_executable(tracking_objects_by_color src/tracking_objects_by_color.cpp)
target_link_libraries(tracking_objects_by_color ${OpenCV_LIBS})

add_executable(triangulation src/triangulation.cpp)
set_property(TARGET triangulation PROPERTY CXX_STANDARD 17)
target_link_libraries(triangulation rerun_sdk ${OpenCV_LIBS})

add_executable(structured_light_range_finding src/structured_light_range_finding.cpp)
target_link_libraries(structured_light_range_finding ${OpenCV_LIBS})

add_executable(convolution src/convolution.cpp)
target_link_libraries(convolution ${OpenCV_LIBS})

add_executable(virtual_camera_calibration src/virtual_camera_calibration.cpp)
target_link_libraries(virtual_camera_calibration ${OpenCV_LIBS})

add_executable(virtual_stereo_vision_cameras src/virtual_stereo_vision_cameras.cpp)
target_link_libraries(virtual_stereo_vision_cameras rerun_sdk ${OpenCV_LIBS})

add_executable(rodrigue_rotation_matrices src/rodrigue_rotation_matrices.cpp)
target_link_libraries(rodrigue_rotation_matrices ${OpenCV_LIBS})

add_executable(finding_homography_using_keypoints_RANSAC src/finding_homography_using_keypoints_RANSAC.cpp)
target_link_libraries(finding_homography_using_keypoints_RANSAC ${OpenCV_LIBS})

add_executable(perspective_transformation src/perspective_transformation.cpp)
target_link_libraries(perspective_transformation ${OpenCV_LIBS})

add_executable(undistorting_images src/undistorting_images.cpp)
target_link_libraries(undistorting_images ${OpenCV_LIBS})

add_executable(rectifying_images src/rectifying_images.cpp)
target_link_libraries(rectifying_images ${OpenCV_LIBS})

add_executable(remap src/remap.cpp)
target_link_libraries(remap ${OpenCV_LIBS})

add_executable(file_storage_io src/file_storage_io.cpp)
target_link_libraries(file_storage_io ${OpenCV_LIBS})

add_executable(camera_projection_matrix src/camera_projection_matrix.cpp)
target_link_libraries(camera_projection_matrix rerun_sdk ${OpenCV_LIBS})

add_executable(finding_homography_matrix_4_corresponding_points src/finding_homography_matrix_4_corresponding_points.cpp)
target_link_libraries(finding_homography_matrix_4_corresponding_points ${OpenCV_LIBS})

add_executable(apply_homography_on_image src/apply_homography_on_image.cpp)
target_link_libraries(apply_homography_on_image ${OpenCV_LIBS})

add_executable(decompose_essential_matrix src/decompose_essential_matrix.cpp)
target_link_libraries(decompose_essential_matrix ${OpenCV_LIBS})

add_executable(decompose_projection_matrix src/decompose_projection_matrix.cpp)
target_link_libraries(decompose_projection_matrix ${OpenCV_LIBS})

add_executable(decompose_homography_matrix src/decompose_homography_matrix.cpp)
target_link_libraries(decompose_homography_matrix ${OpenCV_LIBS})

add_executable(calibrate_hand_eye src/calibrate_hand_eye.cpp)
target_link_libraries(calibrate_hand_eye ${OpenCV_LIBS})

add_executable(computes_camera_characteristics_from_camera_intrinsic src/computes_camera_characteristics_from_camera_intrinsic.cpp)
target_link_libraries(computes_camera_characteristics_from_camera_intrinsic ${OpenCV_LIBS})

add_executable(P3P src/P3P.cpp)
target_link_libraries(P3P rerun_sdk ${OpenCV_LIBS})

add_executable(PnP src/PnP.cpp)
target_link_libraries(PnP ${OpenCV_LIBS})

# add_executable(perspective-n-point src/perspective-n-point.cpp)
# target_link_libraries(perspective-n-point ${OpenCV_LIBS})

add_executable(drawing_frame_axes src/drawing_frame_axes.cpp)
target_link_libraries(drawing_frame_axes ${OpenCV_LIBS})

add_executable(camera_rays src/camera_rays.cpp)
target_link_libraries(camera_rays ${OpenCV_LIBS})



add_executable(undistort_points src/undistort_points.cpp)
target_link_libraries(undistort_points ${OpenCV_LIBS})



find_package(Eigen3 NO_MODULE)

IF(${EIGEN3_FOUND})
	MESSAGE("EIGEN3_FOUND: " ${EIGEN3_FOUND})
	MESSAGE("EIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})
	MESSAGE("EIGEN3_VERSION: " ${EIGEN3_VERSION})
	MESSAGE("EIGEN3_VERSION_STRING: " ${EIGEN3_VERSION_STRING})
	include_directories(${EIGEN3_INCLUDE_DIR})

	add_executable(kalman_filter src/kalman_filter.cpp)
	target_link_libraries(kalman_filter ${OpenCV_LIBS})
ENDIF()

if(EXISTS "${CMAKE_SOURCE_DIR}/src/sandbox.cpp")
	add_executable(sandbox src/sandbox.cpp)
	target_link_libraries(sandbox ${OpenCV_LIBS})
endif()

# add_executable(visual_odometry src/vo/visual_odometry.cpp)
# target_link_libraries(visual_odometry ${OpenCV_LIBS})

# add_executable(visodo src/vo/visodo.cpp)
# target_link_libraries(visodo ${OpenCV_LIBS})
