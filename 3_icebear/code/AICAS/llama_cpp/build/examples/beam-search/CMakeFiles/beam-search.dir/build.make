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
CMAKE_SOURCE_DIR = /root/AICAS/llama_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/AICAS/llama_cpp/build

# Include any dependencies generated for this target.
include examples/beam-search/CMakeFiles/beam-search.dir/depend.make

# Include the progress variables for this target.
include examples/beam-search/CMakeFiles/beam-search.dir/progress.make

# Include the compile flags for this target's objects.
include examples/beam-search/CMakeFiles/beam-search.dir/flags.make

examples/beam-search/CMakeFiles/beam-search.dir/beam-search.cpp.o: examples/beam-search/CMakeFiles/beam-search.dir/flags.make
examples/beam-search/CMakeFiles/beam-search.dir/beam-search.cpp.o: ../examples/beam-search/beam-search.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/AICAS/llama_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/beam-search/CMakeFiles/beam-search.dir/beam-search.cpp.o"
	cd /root/AICAS/llama_cpp/build/examples/beam-search && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/beam-search.dir/beam-search.cpp.o -c /root/AICAS/llama_cpp/examples/beam-search/beam-search.cpp

examples/beam-search/CMakeFiles/beam-search.dir/beam-search.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/beam-search.dir/beam-search.cpp.i"
	cd /root/AICAS/llama_cpp/build/examples/beam-search && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/AICAS/llama_cpp/examples/beam-search/beam-search.cpp > CMakeFiles/beam-search.dir/beam-search.cpp.i

examples/beam-search/CMakeFiles/beam-search.dir/beam-search.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/beam-search.dir/beam-search.cpp.s"
	cd /root/AICAS/llama_cpp/build/examples/beam-search && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/AICAS/llama_cpp/examples/beam-search/beam-search.cpp -o CMakeFiles/beam-search.dir/beam-search.cpp.s

# Object files for target beam-search
beam__search_OBJECTS = \
"CMakeFiles/beam-search.dir/beam-search.cpp.o"

# External object files for target beam-search
beam__search_EXTERNAL_OBJECTS =

bin/beam-search: examples/beam-search/CMakeFiles/beam-search.dir/beam-search.cpp.o
bin/beam-search: examples/beam-search/CMakeFiles/beam-search.dir/build.make
bin/beam-search: common/libcommon.a
bin/beam-search: libllama.a
bin/beam-search: examples/beam-search/CMakeFiles/beam-search.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/AICAS/llama_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/beam-search"
	cd /root/AICAS/llama_cpp/build/examples/beam-search && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/beam-search.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/beam-search/CMakeFiles/beam-search.dir/build: bin/beam-search

.PHONY : examples/beam-search/CMakeFiles/beam-search.dir/build

examples/beam-search/CMakeFiles/beam-search.dir/clean:
	cd /root/AICAS/llama_cpp/build/examples/beam-search && $(CMAKE_COMMAND) -P CMakeFiles/beam-search.dir/cmake_clean.cmake
.PHONY : examples/beam-search/CMakeFiles/beam-search.dir/clean

examples/beam-search/CMakeFiles/beam-search.dir/depend:
	cd /root/AICAS/llama_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/AICAS/llama_cpp /root/AICAS/llama_cpp/examples/beam-search /root/AICAS/llama_cpp/build /root/AICAS/llama_cpp/build/examples/beam-search /root/AICAS/llama_cpp/build/examples/beam-search/CMakeFiles/beam-search.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/beam-search/CMakeFiles/beam-search.dir/depend

