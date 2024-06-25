# Install script for directory: /home/samqiao/ASRL/vtr3_testing/test/src/vtr_testing_radar

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/samqiao/ASRL/vtr3_testing/test/install/vtr_testing_radar")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/samqiao/ASRL/vtr3_testing/test/build/vtr_testing_radar/ament_cmake_symlink_install/ament_cmake_symlink_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_preprocessing" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_preprocessing")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_preprocessing"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar" TYPE EXECUTABLE FILES "/home/samqiao/ASRL/vtr3_testing/test/build/vtr_testing_radar/vtr_testing_radar_radar_preprocessing")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_preprocessing" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_preprocessing")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_preprocessing"
         OLD_RPATH "/usr/local/cuda/lib64:/opt/ros/humble/lib:/home/samqiao/ASRL/vtr3/src/main/install/vtr_radar_msgs/lib:/usr/local/opencv_cuda/lib:/home/samqiao/ASRL/vtr3/src/main/install/navtech_msgs/lib:/home/samqiao/ASRL/vtr3/src/main/install/vtr_tactic_msgs/lib:/home/samqiao/ASRL/vtr3/src/main/install/vtr_storage/lib:/home/samqiao/ASRL/vtr3/src/main/install/vtr_pose_graph_msgs/lib:/home/samqiao/ASRL/vtr3/src/main/install/vtr_common_msgs/lib:/opt/torch/libtorch/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_preprocessing")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_odometry" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_odometry")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_odometry"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar" TYPE EXECUTABLE FILES "/home/samqiao/ASRL/vtr3_testing/test/build/vtr_testing_radar/vtr_testing_radar_radar_odometry")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_odometry" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_odometry")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_odometry"
         OLD_RPATH "/usr/local/cuda/lib64:/opt/ros/humble/lib:/home/samqiao/ASRL/vtr3/src/main/install/vtr_radar_msgs/lib:/usr/local/opencv_cuda/lib:/home/samqiao/ASRL/vtr3/src/main/install/navtech_msgs/lib:/home/samqiao/ASRL/vtr3/src/main/install/vtr_tactic_msgs/lib:/home/samqiao/ASRL/vtr3/src/main/install/vtr_storage/lib:/home/samqiao/ASRL/vtr3/src/main/install/vtr_pose_graph_msgs/lib:/home/samqiao/ASRL/vtr3/src/main/install/vtr_common_msgs/lib:/opt/torch/libtorch/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_odometry")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_localization" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_localization")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_localization"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar" TYPE EXECUTABLE FILES "/home/samqiao/ASRL/vtr3_testing/test/build/vtr_testing_radar/vtr_testing_radar_radar_localization")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_localization" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_localization")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_localization"
         OLD_RPATH "/usr/local/cuda/lib64:/opt/ros/humble/lib:/home/samqiao/ASRL/vtr3/src/main/install/vtr_radar_msgs/lib:/usr/local/opencv_cuda/lib:/home/samqiao/ASRL/vtr3/src/main/install/navtech_msgs/lib:/home/samqiao/ASRL/vtr3/src/main/install/vtr_tactic_msgs/lib:/home/samqiao/ASRL/vtr3/src/main/install/vtr_storage/lib:/home/samqiao/ASRL/vtr3/src/main/install/vtr_pose_graph_msgs/lib:/home/samqiao/ASRL/vtr3/src/main/install/vtr_common_msgs/lib:/opt/torch/libtorch/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/vtr_testing_radar/vtr_testing_radar_radar_localization")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/samqiao/ASRL/vtr3_testing/test/build/vtr_testing_radar/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
