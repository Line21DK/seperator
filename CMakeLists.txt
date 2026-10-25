cmake_minimum_required(VERSION 3.20)

if(APPLE)
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "")
  set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "")
endif()

project(Seperator VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
# OpenFX headers and the C++ Support library, pinned for reproducible builds.
FetchContent_Declare(openfx
  GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openfx.git
  GIT_TAG c3f1c2dcf4f51762b794c4fc21181b4a501be1bf
  GIT_SHALLOW FALSE
  SOURCE_SUBDIR do-not-build)  # headers/sources only, skip OpenFX's own CMake
FetchContent_MakeAvailable(openfx)

find_package(Threads REQUIRED)

add_library(seperator_core STATIC src/core/FreqSep.cpp)
target_include_directories(seperator_core PUBLIC src/core)
target_link_libraries(seperator_core PUBLIC Threads::Threads)

add_executable(sep_cli tools/sep_cli.cpp)
target_link_libraries(sep_cli PRIVATE seperator_core)

file(GLOB OFX_SUPPORT_SOURCES ${openfx_SOURCE_DIR}/Support/Library/*.cpp)
add_library(Seperator MODULE src/ofx/SeperatorPlugin.cpp ${OFX_SUPPORT_SOURCES})
target_include_directories(Seperator PRIVATE
  ${openfx_SOURCE_DIR}/include
  ${openfx_SOURCE_DIR}/Support/include
  ${openfx_SOURCE_DIR}/Support/Library)
target_link_libraries(Seperator PRIVATE seperator_core)
set_target_properties(Seperator PROPERTIES PREFIX "" SUFFIX ".ofx"
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

# Lay the plugin out as an OFX bundle: Seperator.ofx.bundle/Contents/<arch>/Seperator.ofx
if(APPLE)
  set(OFX_ARCH_DIR MacOS)
elseif(WIN32)
  set(OFX_ARCH_DIR Win64)
else()
  set(OFX_ARCH_DIR Linux-x86-64)
endif()
set(OFX_BUNDLE_DIR ${CMAKE_BINARY_DIR}/Seperator.ofx.bundle)
configure_file(resources/Info.plist.in ${OFX_BUNDLE_DIR}/Contents/Info.plist @ONLY)
add_custom_command(TARGET Seperator POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OFX_BUNDLE_DIR}/Contents/${OFX_ARCH_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:Seperator> ${OFX_BUNDLE_DIR}/Contents/${OFX_ARCH_DIR}/
  COMMENT "Assembling Seperator.ofx.bundle")

install(DIRECTORY ${OFX_BUNDLE_DIR} DESTINATION .)

enable_testing()
add_executable(core_test tests/core_test.cpp)
target_link_libraries(core_test PRIVATE seperator_core)
add_test(NAME core_test COMMAND core_test)
