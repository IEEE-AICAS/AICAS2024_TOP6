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
include CMakeFiles/llama.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/llama.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/llama.dir/flags.make

CMakeFiles/llama.dir/llama.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/llama.cpp.o: ../llama.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/AICAS/llama_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/llama.dir/llama.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/llama.dir/llama.cpp.o -c /root/AICAS/llama_cpp/llama.cpp

CMakeFiles/llama.dir/llama.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/llama.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/AICAS/llama_cpp/llama.cpp > CMakeFiles/llama.dir/llama.cpp.i

CMakeFiles/llama.dir/llama.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/llama.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/AICAS/llama_cpp/llama.cpp -o CMakeFiles/llama.dir/llama.cpp.s

CMakeFiles/llama.dir/unicode.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/unicode.cpp.o: ../unicode.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/AICAS/llama_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/llama.dir/unicode.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/llama.dir/unicode.cpp.o -c /root/AICAS/llama_cpp/unicode.cpp

CMakeFiles/llama.dir/unicode.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/unicode.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/AICAS/llama_cpp/unicode.cpp > CMakeFiles/llama.dir/unicode.cpp.i

CMakeFiles/llama.dir/unicode.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/unicode.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/AICAS/llama_cpp/unicode.cpp -o CMakeFiles/llama.dir/unicode.cpp.s

CMakeFiles/llama.dir/unicode-data.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/unicode-data.cpp.o: ../unicode-data.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/AICAS/llama_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/llama.dir/unicode-data.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/llama.dir/unicode-data.cpp.o -c /root/AICAS/llama_cpp/unicode-data.cpp

CMakeFiles/llama.dir/unicode-data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/unicode-data.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/AICAS/llama_cpp/unicode-data.cpp > CMakeFiles/llama.dir/unicode-data.cpp.i

CMakeFiles/llama.dir/unicode-data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/unicode-data.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/AICAS/llama_cpp/unicode-data.cpp -o CMakeFiles/llama.dir/unicode-data.cpp.s

# Object files for target llama
llama_OBJECTS = \
"CMakeFiles/llama.dir/llama.cpp.o" \
"CMakeFiles/llama.dir/unicode.cpp.o" \
"CMakeFiles/llama.dir/unicode-data.cpp.o"

# External object files for target llama
llama_EXTERNAL_OBJECTS = \
"/root/AICAS/llama_cpp/build/CMakeFiles/ggml.dir/ggml.c.o" \
"/root/AICAS/llama_cpp/build/CMakeFiles/ggml.dir/ggml-alloc.c.o" \
"/root/AICAS/llama_cpp/build/CMakeFiles/ggml.dir/ggml-backend.c.o" \
"/root/AICAS/llama_cpp/build/CMakeFiles/ggml.dir/ggml-quants.c.o" \
"/root/AICAS/llama_cpp/build/CMakeFiles/ggml.dir/sgemm.cpp.o"

libllama.a: CMakeFiles/llama.dir/llama.cpp.o
libllama.a: CMakeFiles/llama.dir/unicode.cpp.o
libllama.a: CMakeFiles/llama.dir/unicode-data.cpp.o
libllama.a: CMakeFiles/ggml.dir/ggml.c.o
libllama.a: CMakeFiles/ggml.dir/ggml-alloc.c.o
libllama.a: CMakeFiles/ggml.dir/ggml-backend.c.o
libllama.a: CMakeFiles/ggml.dir/ggml-quants.c.o
libllama.a: CMakeFiles/ggml.dir/sgemm.cpp.o
libllama.a: CMakeFiles/llama.dir/build.make
libllama.a: CMakeFiles/llama.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/AICAS/llama_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libllama.a"
	$(CMAKE_COMMAND) -P CMakeFiles/llama.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/llama.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/llama.dir/build: libllama.a

.PHONY : CMakeFiles/llama.dir/build

CMakeFiles/llama.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/llama.dir/cmake_clean.cmake
.PHONY : CMakeFiles/llama.dir/clean

CMakeFiles/llama.dir/depend:
	cd /root/AICAS/llama_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/AICAS/llama_cpp /root/AICAS/llama_cpp /root/AICAS/llama_cpp/build /root/AICAS/llama_cpp/build /root/AICAS/llama_cpp/build/CMakeFiles/llama.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/llama.dir/depend
