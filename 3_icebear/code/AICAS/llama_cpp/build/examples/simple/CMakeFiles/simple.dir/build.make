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
include examples/simple/CMakeFiles/simple.dir/depend.make

# Include the progress variables for this target.
include examples/simple/CMakeFiles/simple.dir/progress.make

# Include the compile flags for this target's objects.
include examples/simple/CMakeFiles/simple.dir/flags.make

examples/simple/CMakeFiles/simple.dir/simple.cpp.o: examples/simple/CMakeFiles/simple.dir/flags.make
examples/simple/CMakeFiles/simple.dir/simple.cpp.o: ../examples/simple/simple.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/AICAS/llama_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/simple/CMakeFiles/simple.dir/simple.cpp.o"
	cd /root/AICAS/llama_cpp/build/examples/simple && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/simple.dir/simple.cpp.o -c /root/AICAS/llama_cpp/examples/simple/simple.cpp

examples/simple/CMakeFiles/simple.dir/simple.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simple.dir/simple.cpp.i"
	cd /root/AICAS/llama_cpp/build/examples/simple && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/AICAS/llama_cpp/examples/simple/simple.cpp > CMakeFiles/simple.dir/simple.cpp.i

examples/simple/CMakeFiles/simple.dir/simple.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simple.dir/simple.cpp.s"
	cd /root/AICAS/llama_cpp/build/examples/simple && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/AICAS/llama_cpp/examples/simple/simple.cpp -o CMakeFiles/simple.dir/simple.cpp.s

# Object files for target simple
simple_OBJECTS = \
"CMakeFiles/simple.dir/simple.cpp.o"

# External object files for target simple
simple_EXTERNAL_OBJECTS =

bin/simple: examples/simple/CMakeFiles/simple.dir/simple.cpp.o
bin/simple: examples/simple/CMakeFiles/simple.dir/build.make
bin/simple: common/libcommon.a
bin/simple: libllama.a
bin/simple: examples/simple/CMakeFiles/simple.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/AICAS/llama_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/simple"
	cd /root/AICAS/llama_cpp/build/examples/simple && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simple.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/simple/CMakeFiles/simple.dir/build: bin/simple

.PHONY : examples/simple/CMakeFiles/simple.dir/build

examples/simple/CMakeFiles/simple.dir/clean:
	cd /root/AICAS/llama_cpp/build/examples/simple && $(CMAKE_COMMAND) -P CMakeFiles/simple.dir/cmake_clean.cmake
.PHONY : examples/simple/CMakeFiles/simple.dir/clean

examples/simple/CMakeFiles/simple.dir/depend:
	cd /root/AICAS/llama_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/AICAS/llama_cpp /root/AICAS/llama_cpp/examples/simple /root/AICAS/llama_cpp/build /root/AICAS/llama_cpp/build/examples/simple /root/AICAS/llama_cpp/build/examples/simple/CMakeFiles/simple.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/simple/CMakeFiles/simple.dir/depend
