# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhouhongjin/第二次任务喵/task

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhouhongjin/第二次任务喵/task/build

# Include any dependencies generated for this target.
include CMakeFiles/task.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/task.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/task.dir/flags.make

CMakeFiles/task.dir/main.cpp.o: CMakeFiles/task.dir/flags.make
CMakeFiles/task.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhouhongjin/第二次任务喵/task/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/task.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/task.dir/main.cpp.o -c /home/zhouhongjin/第二次任务喵/task/main.cpp

CMakeFiles/task.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/task.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhouhongjin/第二次任务喵/task/main.cpp > CMakeFiles/task.dir/main.cpp.i

CMakeFiles/task.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/task.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhouhongjin/第二次任务喵/task/main.cpp -o CMakeFiles/task.dir/main.cpp.s

CMakeFiles/task.dir/src/windmill.cpp.o: CMakeFiles/task.dir/flags.make
CMakeFiles/task.dir/src/windmill.cpp.o: ../src/windmill.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhouhongjin/第二次任务喵/task/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/task.dir/src/windmill.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/task.dir/src/windmill.cpp.o -c /home/zhouhongjin/第二次任务喵/task/src/windmill.cpp

CMakeFiles/task.dir/src/windmill.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/task.dir/src/windmill.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhouhongjin/第二次任务喵/task/src/windmill.cpp > CMakeFiles/task.dir/src/windmill.cpp.i

CMakeFiles/task.dir/src/windmill.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/task.dir/src/windmill.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhouhongjin/第二次任务喵/task/src/windmill.cpp -o CMakeFiles/task.dir/src/windmill.cpp.s

# Object files for target task
task_OBJECTS = \
"CMakeFiles/task.dir/main.cpp.o" \
"CMakeFiles/task.dir/src/windmill.cpp.o"

# External object files for target task
task_EXTERNAL_OBJECTS =

task: CMakeFiles/task.dir/main.cpp.o
task: CMakeFiles/task.dir/src/windmill.cpp.o
task: CMakeFiles/task.dir/build.make
task: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d
task: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d
task: CMakeFiles/task.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhouhongjin/第二次任务喵/task/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable task"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/task.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/task.dir/build: task

.PHONY : CMakeFiles/task.dir/build

CMakeFiles/task.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/task.dir/cmake_clean.cmake
.PHONY : CMakeFiles/task.dir/clean

CMakeFiles/task.dir/depend:
	cd /home/zhouhongjin/第二次任务喵/task/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhouhongjin/第二次任务喵/task /home/zhouhongjin/第二次任务喵/task /home/zhouhongjin/第二次任务喵/task/build /home/zhouhongjin/第二次任务喵/task/build /home/zhouhongjin/第二次任务喵/task/build/CMakeFiles/task.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/task.dir/depend

