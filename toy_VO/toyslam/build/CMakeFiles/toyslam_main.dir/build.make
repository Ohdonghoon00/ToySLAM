# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/donghoon/ToySLAM/toy_VO/toyslam

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/donghoon/ToySLAM/toy_VO/toyslam/build

# Include any dependencies generated for this target.
include CMakeFiles/toyslam_main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/toyslam_main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/toyslam_main.dir/flags.make

CMakeFiles/toyslam_main.dir/toyslam_main.cpp.o: CMakeFiles/toyslam_main.dir/flags.make
CMakeFiles/toyslam_main.dir/toyslam_main.cpp.o: ../toyslam_main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/donghoon/ToySLAM/toy_VO/toyslam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/toyslam_main.dir/toyslam_main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/toyslam_main.dir/toyslam_main.cpp.o -c /home/donghoon/ToySLAM/toy_VO/toyslam/toyslam_main.cpp

CMakeFiles/toyslam_main.dir/toyslam_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/toyslam_main.dir/toyslam_main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/donghoon/ToySLAM/toy_VO/toyslam/toyslam_main.cpp > CMakeFiles/toyslam_main.dir/toyslam_main.cpp.i

CMakeFiles/toyslam_main.dir/toyslam_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/toyslam_main.dir/toyslam_main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/donghoon/ToySLAM/toy_VO/toyslam/toyslam_main.cpp -o CMakeFiles/toyslam_main.dir/toyslam_main.cpp.s

CMakeFiles/toyslam_main.dir/toyslam_main.cpp.o.requires:

.PHONY : CMakeFiles/toyslam_main.dir/toyslam_main.cpp.o.requires

CMakeFiles/toyslam_main.dir/toyslam_main.cpp.o.provides: CMakeFiles/toyslam_main.dir/toyslam_main.cpp.o.requires
	$(MAKE) -f CMakeFiles/toyslam_main.dir/build.make CMakeFiles/toyslam_main.dir/toyslam_main.cpp.o.provides.build
.PHONY : CMakeFiles/toyslam_main.dir/toyslam_main.cpp.o.provides

CMakeFiles/toyslam_main.dir/toyslam_main.cpp.o.provides.build: CMakeFiles/toyslam_main.dir/toyslam_main.cpp.o


CMakeFiles/toyslam_main.dir/math.cpp.o: CMakeFiles/toyslam_main.dir/flags.make
CMakeFiles/toyslam_main.dir/math.cpp.o: ../math.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/donghoon/ToySLAM/toy_VO/toyslam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/toyslam_main.dir/math.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/toyslam_main.dir/math.cpp.o -c /home/donghoon/ToySLAM/toy_VO/toyslam/math.cpp

CMakeFiles/toyslam_main.dir/math.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/toyslam_main.dir/math.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/donghoon/ToySLAM/toy_VO/toyslam/math.cpp > CMakeFiles/toyslam_main.dir/math.cpp.i

CMakeFiles/toyslam_main.dir/math.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/toyslam_main.dir/math.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/donghoon/ToySLAM/toy_VO/toyslam/math.cpp -o CMakeFiles/toyslam_main.dir/math.cpp.s

CMakeFiles/toyslam_main.dir/math.cpp.o.requires:

.PHONY : CMakeFiles/toyslam_main.dir/math.cpp.o.requires

CMakeFiles/toyslam_main.dir/math.cpp.o.provides: CMakeFiles/toyslam_main.dir/math.cpp.o.requires
	$(MAKE) -f CMakeFiles/toyslam_main.dir/build.make CMakeFiles/toyslam_main.dir/math.cpp.o.provides.build
.PHONY : CMakeFiles/toyslam_main.dir/math.cpp.o.provides

CMakeFiles/toyslam_main.dir/math.cpp.o.provides.build: CMakeFiles/toyslam_main.dir/math.cpp.o


# Object files for target toyslam_main
toyslam_main_OBJECTS = \
"CMakeFiles/toyslam_main.dir/toyslam_main.cpp.o" \
"CMakeFiles/toyslam_main.dir/math.cpp.o"

# External object files for target toyslam_main
toyslam_main_EXTERNAL_OBJECTS =

toyslam_main: CMakeFiles/toyslam_main.dir/toyslam_main.cpp.o
toyslam_main: CMakeFiles/toyslam_main.dir/math.cpp.o
toyslam_main: CMakeFiles/toyslam_main.dir/build.make
toyslam_main: types/libtoyslam_types_library.a
toyslam_main: feature/libtoyslam_feature_library.a
toyslam_main: track/libtoyslam_track_library.a
toyslam_main: bundleAdjustment/libtoyslam_bundleadjustment_library.a
toyslam_main: graph_optimization/libtoyslam_graph_optimization_library.a
toyslam_main: visualization/libtoyslam_visualization_library.a
toyslam_main: /usr/local/lib/libceres.a
toyslam_main: /usr/lib/x86_64-linux-gnu/libglog.so
toyslam_main: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1
toyslam_main: /usr/lib/x86_64-linux-gnu/libspqr.so
toyslam_main: /usr/lib/x86_64-linux-gnu/libtbb.so
toyslam_main: /usr/lib/x86_64-linux-gnu/libcholmod.so
toyslam_main: /usr/lib/x86_64-linux-gnu/libccolamd.so
toyslam_main: /usr/lib/x86_64-linux-gnu/libcamd.so
toyslam_main: /usr/lib/x86_64-linux-gnu/libcolamd.so
toyslam_main: /usr/lib/x86_64-linux-gnu/libamd.so
toyslam_main: /usr/lib/x86_64-linux-gnu/liblapack.so
toyslam_main: /usr/lib/x86_64-linux-gnu/libf77blas.so
toyslam_main: /usr/lib/x86_64-linux-gnu/libatlas.so
toyslam_main: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
toyslam_main: /usr/lib/x86_64-linux-gnu/librt.so
toyslam_main: /usr/lib/x86_64-linux-gnu/libcxsparse.so
toyslam_main: /usr/lib/x86_64-linux-gnu/liblapack.so
toyslam_main: /usr/lib/x86_64-linux-gnu/libf77blas.so
toyslam_main: /usr/lib/x86_64-linux-gnu/libatlas.so
toyslam_main: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
toyslam_main: /usr/lib/x86_64-linux-gnu/librt.so
toyslam_main: /usr/lib/x86_64-linux-gnu/libcxsparse.so
toyslam_main: graph_optimization/libDBoW2.so
toyslam_main: /usr/local/lib/libopencv_gapi.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_stitching.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_aruco.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_bgsegm.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_bioinspired.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_ccalib.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_dnn_objdetect.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_dnn_superres.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_dpm.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_highgui.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_face.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_freetype.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_fuzzy.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_hdf.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_hfs.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_img_hash.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_line_descriptor.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_quality.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_reg.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_rgbd.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_saliency.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_sfm.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_stereo.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_structured_light.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_phase_unwrapping.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_superres.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_optflow.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_surface_matching.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_tracking.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_datasets.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_plot.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_text.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_dnn.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_videostab.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_videoio.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_xfeatures2d.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_ml.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_shape.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_ximgproc.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_video.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_xobjdetect.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_imgcodecs.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_objdetect.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_calib3d.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_features2d.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_flann.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_xphoto.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_photo.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_imgproc.so.4.2.0
toyslam_main: /usr/local/lib/libopencv_core.so.4.2.0
toyslam_main: CMakeFiles/toyslam_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/donghoon/ToySLAM/toy_VO/toyslam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable toyslam_main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/toyslam_main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/toyslam_main.dir/build: toyslam_main

.PHONY : CMakeFiles/toyslam_main.dir/build

CMakeFiles/toyslam_main.dir/requires: CMakeFiles/toyslam_main.dir/toyslam_main.cpp.o.requires
CMakeFiles/toyslam_main.dir/requires: CMakeFiles/toyslam_main.dir/math.cpp.o.requires

.PHONY : CMakeFiles/toyslam_main.dir/requires

CMakeFiles/toyslam_main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/toyslam_main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/toyslam_main.dir/clean

CMakeFiles/toyslam_main.dir/depend:
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/donghoon/ToySLAM/toy_VO/toyslam /home/donghoon/ToySLAM/toy_VO/toyslam /home/donghoon/ToySLAM/toy_VO/toyslam/build /home/donghoon/ToySLAM/toy_VO/toyslam/build /home/donghoon/ToySLAM/toy_VO/toyslam/build/CMakeFiles/toyslam_main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/toyslam_main.dir/depend
