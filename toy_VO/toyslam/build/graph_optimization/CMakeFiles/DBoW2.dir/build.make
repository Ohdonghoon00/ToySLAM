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
include graph_optimization/CMakeFiles/DBoW2.dir/depend.make

# Include the progress variables for this target.
include graph_optimization/CMakeFiles/DBoW2.dir/progress.make

# Include the compile flags for this target's objects.
include graph_optimization/CMakeFiles/DBoW2.dir/flags.make

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.o: graph_optimization/CMakeFiles/DBoW2.dir/flags.make
graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.o: ../graph_optimization/DBoW2/src/BowVector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/donghoon/ToySLAM/toy_VO/toyslam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.o"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.o -c /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/BowVector.cpp

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.i"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/BowVector.cpp > CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.i

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.s"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/BowVector.cpp -o CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.s

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.o.requires:

.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.o.requires

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.o.provides: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.o.requires
	$(MAKE) -f graph_optimization/CMakeFiles/DBoW2.dir/build.make graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.o.provides.build
.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.o.provides

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.o.provides.build: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.o


graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.o: graph_optimization/CMakeFiles/DBoW2.dir/flags.make
graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.o: ../graph_optimization/DBoW2/src/FBrief.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/donghoon/ToySLAM/toy_VO/toyslam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.o"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.o -c /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/FBrief.cpp

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.i"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/FBrief.cpp > CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.i

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.s"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/FBrief.cpp -o CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.s

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.o.requires:

.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.o.requires

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.o.provides: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.o.requires
	$(MAKE) -f graph_optimization/CMakeFiles/DBoW2.dir/build.make graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.o.provides.build
.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.o.provides

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.o.provides.build: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.o


graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.o: graph_optimization/CMakeFiles/DBoW2.dir/flags.make
graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.o: ../graph_optimization/DBoW2/src/FORB.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/donghoon/ToySLAM/toy_VO/toyslam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.o"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.o -c /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/FORB.cpp

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.i"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/FORB.cpp > CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.i

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.s"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/FORB.cpp -o CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.s

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.o.requires:

.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.o.requires

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.o.provides: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.o.requires
	$(MAKE) -f graph_optimization/CMakeFiles/DBoW2.dir/build.make graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.o.provides.build
.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.o.provides

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.o.provides.build: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.o


graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.o: graph_optimization/CMakeFiles/DBoW2.dir/flags.make
graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.o: ../graph_optimization/DBoW2/src/FeatureVector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/donghoon/ToySLAM/toy_VO/toyslam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.o"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.o -c /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/FeatureVector.cpp

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.i"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/FeatureVector.cpp > CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.i

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.s"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/FeatureVector.cpp -o CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.s

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.o.requires:

.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.o.requires

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.o.provides: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.o.requires
	$(MAKE) -f graph_optimization/CMakeFiles/DBoW2.dir/build.make graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.o.provides.build
.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.o.provides

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.o.provides.build: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.o


graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.o: graph_optimization/CMakeFiles/DBoW2.dir/flags.make
graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.o: ../graph_optimization/DBoW2/src/QueryResults.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/donghoon/ToySLAM/toy_VO/toyslam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.o"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.o -c /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/QueryResults.cpp

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.i"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/QueryResults.cpp > CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.i

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.s"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/QueryResults.cpp -o CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.s

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.o.requires:

.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.o.requires

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.o.provides: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.o.requires
	$(MAKE) -f graph_optimization/CMakeFiles/DBoW2.dir/build.make graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.o.provides.build
.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.o.provides

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.o.provides.build: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.o


graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.o: graph_optimization/CMakeFiles/DBoW2.dir/flags.make
graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.o: ../graph_optimization/DBoW2/src/ScoringObject.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/donghoon/ToySLAM/toy_VO/toyslam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.o"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.o -c /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/ScoringObject.cpp

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.i"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/ScoringObject.cpp > CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.i

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.s"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization/DBoW2/src/ScoringObject.cpp -o CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.s

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.o.requires:

.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.o.requires

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.o.provides: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.o.requires
	$(MAKE) -f graph_optimization/CMakeFiles/DBoW2.dir/build.make graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.o.provides.build
.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.o.provides

graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.o.provides.build: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.o


# Object files for target DBoW2
DBoW2_OBJECTS = \
"CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.o" \
"CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.o" \
"CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.o" \
"CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.o" \
"CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.o" \
"CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.o"

# External object files for target DBoW2
DBoW2_EXTERNAL_OBJECTS =

graph_optimization/libDBoW2.so: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.o
graph_optimization/libDBoW2.so: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.o
graph_optimization/libDBoW2.so: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.o
graph_optimization/libDBoW2.so: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.o
graph_optimization/libDBoW2.so: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.o
graph_optimization/libDBoW2.so: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.o
graph_optimization/libDBoW2.so: graph_optimization/CMakeFiles/DBoW2.dir/build.make
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_gapi.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_stitching.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_aruco.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_bgsegm.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_bioinspired.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_ccalib.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_dnn_objdetect.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_dnn_superres.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_dpm.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_face.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_freetype.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_fuzzy.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_hdf.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_hfs.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_img_hash.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_line_descriptor.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_quality.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_reg.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_rgbd.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_saliency.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_sfm.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_stereo.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_structured_light.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_superres.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_surface_matching.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_tracking.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_videostab.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_xfeatures2d.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_xobjdetect.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_xphoto.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_highgui.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_shape.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_datasets.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_plot.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_text.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_dnn.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_ml.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_phase_unwrapping.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_optflow.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_ximgproc.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_video.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_videoio.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_imgcodecs.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_objdetect.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_calib3d.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_features2d.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_flann.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_photo.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_imgproc.so.4.2.0
graph_optimization/libDBoW2.so: /usr/local/lib/libopencv_core.so.4.2.0
graph_optimization/libDBoW2.so: graph_optimization/CMakeFiles/DBoW2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/donghoon/ToySLAM/toy_VO/toyslam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX shared library libDBoW2.so"
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DBoW2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
graph_optimization/CMakeFiles/DBoW2.dir/build: graph_optimization/libDBoW2.so

.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/build

graph_optimization/CMakeFiles/DBoW2.dir/requires: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/BowVector.cpp.o.requires
graph_optimization/CMakeFiles/DBoW2.dir/requires: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FBrief.cpp.o.requires
graph_optimization/CMakeFiles/DBoW2.dir/requires: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FORB.cpp.o.requires
graph_optimization/CMakeFiles/DBoW2.dir/requires: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/FeatureVector.cpp.o.requires
graph_optimization/CMakeFiles/DBoW2.dir/requires: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/QueryResults.cpp.o.requires
graph_optimization/CMakeFiles/DBoW2.dir/requires: graph_optimization/CMakeFiles/DBoW2.dir/DBoW2/src/ScoringObject.cpp.o.requires

.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/requires

graph_optimization/CMakeFiles/DBoW2.dir/clean:
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization && $(CMAKE_COMMAND) -P CMakeFiles/DBoW2.dir/cmake_clean.cmake
.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/clean

graph_optimization/CMakeFiles/DBoW2.dir/depend:
	cd /home/donghoon/ToySLAM/toy_VO/toyslam/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/donghoon/ToySLAM/toy_VO/toyslam /home/donghoon/ToySLAM/toy_VO/toyslam/graph_optimization /home/donghoon/ToySLAM/toy_VO/toyslam/build /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization /home/donghoon/ToySLAM/toy_VO/toyslam/build/graph_optimization/CMakeFiles/DBoW2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : graph_optimization/CMakeFiles/DBoW2.dir/depend
