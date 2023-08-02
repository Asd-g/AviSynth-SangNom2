cmake_minimum_required(VERSION 3.16)

project(libsangnom2 LANGUAGES CXX)

set(sources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SangNom2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SangNom2_SSE2.cpp
)

if (WIN32)
    set(sources ${sources} ${CMAKE_CURRENT_SOURCE_DIR}/src/SangNom2.rc)
endif ()

add_library(sangnom2 SHARED ${sources})

if (UNIX)
    target_include_directories(sangnom2 PRIVATE /usr/local/include/avisynth)
else ()
    target_include_directories(sangnom2 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../AviSynthPlus/avs_core/include")
endif ()

if (NOT CMAKE_GENERATOR MATCHES "Visual Studio")
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    endif()

    string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
    if (build_type STREQUAL debug)
        target_compile_definitions(sangnom2 PRIVATE DEBUG_BUILD)
    else (build_type STREQUAL release)
        target_compile_definitions(sangnom2 PRIVATE RELEASE_BUILD)
    endif()

    message(STATUS "Build type - ${CMAKE_BUILD_TYPE}")
endif()

target_compile_features(sangnom2 PRIVATE cxx_std_17)

if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_link_libraries(sangnom2 PRIVATE
        libmmds
        svml_disp
    )
endif ()

if (MINGW)
    target_link_libraries(sangnom2 PRIVATE -static-libstdc++ -static-libgcc -s)
endif()

if (WIN32)
    set_target_properties(sangnom2 PROPERTIES
        PREFIX ""
        OUTPUT_NAME "SangNom2"
    )
else ()
    find_package (Git)    
    if (GIT_FOUND)
        execute_process (COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
            OUTPUT_VARIABLE ver
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    else ()
        message (STATUS "GIT not found")
    endif ()

    set_target_properties(sangnom2 PROPERTIES OUTPUT_NAME "sangnom2.${ver}")
endif ()

if (UNIX)
    include(GNUInstallDirs)

    INSTALL(TARGETS sangnom2 LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth")

    # uninstall target
    if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
    endif ()
endif ()
