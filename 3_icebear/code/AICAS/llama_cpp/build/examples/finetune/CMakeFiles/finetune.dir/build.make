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
include examples/finetune/CMakeFiles/finetune.dir/depend.make

# Include the progress variables for this target.
include examples/finetune/CMakeFiles/finetune.dir/progress.make

# Include the compile flags for this target's objects.
include examples/finetune/CMakeFiles/finetune.dir/flags.make

examples/finetune/CMakeFiles/finetune.dir/finetune.cpp.o: examples/finetune/CMakeFiles/finetune.dir/flags.make
examples/finetune/CMakeFiles/finetune.dir/finetune.cpp.o: ../examples/finetune/finetune.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/AICAS/llama_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/finetune/CMakeFiles/finetune.dir/finetune.cpp.o"
	cd /root/AICAS/llama_cpp/build/examples/finetune && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/finetune.dir/finetune.cpp.o -c /root/AICAS/llama_cpp/examples/finetune/finetune.cpp

examples/finetune/CMakeFiles/finetune.dir/finetune.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/finetune.dir/finetune.cpp.i"
	cd /root/AICAS/llama_cpp/build/examples/finetune && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/AICAS/llama_cpp/examples/finetune/finetune.cpp > CMakeFiles/finetune.dir/finetune.cpp.i

examples/finetune/CMakeFiles/finetune.dir/finetune.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/finetune.dir/finetune.cpp.s"
	cd /root/AICAS/llama_cpp/build/examples/finetune && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/AICAS/llama_cpp/examples/finetune/finetune.cpp -o CMakeFiles/finetune.dir/finetune.cpp.s

# Object files for target finetune
finetune_OBJECTS = \
"CMakeFiles/finetune.dir/finetune.cpp.o"

# External object files for target finetune
finetune_EXTERNAL_OBJECTS =

bin/finetune: examples/finetune/CMakeFiles/finetune.dir/finetune.cpp.o
bin/finetune: examples/finetune/CMakeFiles/finetune.dir/build.make
bin/finetune: common/libcommon.a
bin/finetune: libllama.a
bin/finetune: examples/finetune/CMakeFiles/finetune.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/AICAS/llama_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/finetune"
	cd /root/AICAS/llama_cpp/build/examples/finetune && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/finetune.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/finetune/CMakeFiles/finetune.dir/build: bin/finetune

.PHONY : examples/finetune/CMakeFiles/finetune.dir/build

examples/finetune/CMakeFiles/finetune.dir/clean:
	cd /root/AICAS/llama_cpp/build/examples/finetune && $(CMAKE_COMMAND) -P CMakeFiles/finetune.dir/cmake_clean.cmake
.PHONY : examples/finetune/CMakeFiles/finetune.dir/clean

examples/finetune/CMakeFiles/finetune.dir/depend:
	cd /root/AICAS/llama_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/AICAS/llama_cpp /root/AICAS/llama_cpp/examples/finetune /root/AICAS/llama_cpp/build /root/AICAS/llama_cpp/build/examples/finetune /root/AICAS/llama_cpp/build/examples/finetune/CMakeFiles/finetune.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/finetune/CMakeFiles/finetune.dir/depend
