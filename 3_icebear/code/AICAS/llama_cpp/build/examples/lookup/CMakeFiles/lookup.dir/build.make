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
include examples/lookup/CMakeFiles/lookup.dir/depend.make

# Include the progress variables for this target.
include examples/lookup/CMakeFiles/lookup.dir/progress.make

# Include the compile flags for this target's objects.
include examples/lookup/CMakeFiles/lookup.dir/flags.make

examples/lookup/CMakeFiles/lookup.dir/lookup.cpp.o: examples/lookup/CMakeFiles/lookup.dir/flags.make
examples/lookup/CMakeFiles/lookup.dir/lookup.cpp.o: ../examples/lookup/lookup.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/AICAS/llama_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/lookup/CMakeFiles/lookup.dir/lookup.cpp.o"
	cd /root/AICAS/llama_cpp/build/examples/lookup && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lookup.dir/lookup.cpp.o -c /root/AICAS/llama_cpp/examples/lookup/lookup.cpp

examples/lookup/CMakeFiles/lookup.dir/lookup.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lookup.dir/lookup.cpp.i"
	cd /root/AICAS/llama_cpp/build/examples/lookup && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/AICAS/llama_cpp/examples/lookup/lookup.cpp > CMakeFiles/lookup.dir/lookup.cpp.i

examples/lookup/CMakeFiles/lookup.dir/lookup.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lookup.dir/lookup.cpp.s"
	cd /root/AICAS/llama_cpp/build/examples/lookup && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/AICAS/llama_cpp/examples/lookup/lookup.cpp -o CMakeFiles/lookup.dir/lookup.cpp.s

# Object files for target lookup
lookup_OBJECTS = \
"CMakeFiles/lookup.dir/lookup.cpp.o"

# External object files for target lookup
lookup_EXTERNAL_OBJECTS =

bin/lookup: examples/lookup/CMakeFiles/lookup.dir/lookup.cpp.o
bin/lookup: examples/lookup/CMakeFiles/lookup.dir/build.make
bin/lookup: common/libcommon.a
bin/lookup: libllama.a
bin/lookup: examples/lookup/CMakeFiles/lookup.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/AICAS/llama_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/lookup"
	cd /root/AICAS/llama_cpp/build/examples/lookup && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lookup.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/lookup/CMakeFiles/lookup.dir/build: bin/lookup

.PHONY : examples/lookup/CMakeFiles/lookup.dir/build

examples/lookup/CMakeFiles/lookup.dir/clean:
	cd /root/AICAS/llama_cpp/build/examples/lookup && $(CMAKE_COMMAND) -P CMakeFiles/lookup.dir/cmake_clean.cmake
.PHONY : examples/lookup/CMakeFiles/lookup.dir/clean

examples/lookup/CMakeFiles/lookup.dir/depend:
	cd /root/AICAS/llama_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/AICAS/llama_cpp /root/AICAS/llama_cpp/examples/lookup /root/AICAS/llama_cpp/build /root/AICAS/llama_cpp/build/examples/lookup /root/AICAS/llama_cpp/build/examples/lookup/CMakeFiles/lookup.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/lookup/CMakeFiles/lookup.dir/depend
