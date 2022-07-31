cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME HairRecon)
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.0.1 DESCRIPTION "Hair Reconstruction")

set(CMAKE_VERBOSE_MAKEFILE TRUE)

# .lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# .dll and .exe
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin) 

if (WIN32)
# option for Visual Studio
# -EHsc (enable proper Exxeption Handling) needs to avoid C4530
# -Wall is too noisy so that set -W4. 
# https://docs.microsoft.com/en-us/cpp/build/reference/compiler-option-warning-level?view=vs-2017
# "However, for a new project, it may be best to use /W4 in all compilations;
# this will ensure the fewest possible hard-to-find code defects."
set(CMAKE_CXX_FLAGS "/std:c++17 -W4 -EHsc /bigobj")

else()
# g++ option for *nix
set(CMAKE_CXX_FLAGS  "-std=c++17 -Wall -O2")

endif()

get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
  message(STATUS "Has a parent scope.")
else()
  message(STATUS "Doesn't have a parent scope.")
endif()

function(set_with_parent ARG_NAME ARG_PATH ARG_TEXT)
    set(${ARG_NAME} ${ARG_PATH} CACHE PATH ${ARG_TEXT} FORCE)
    if (hasParent)
      set(${ARG_NAME} ${ARG_PATH} PARENT_SCOPE)
    endif()
endfunction(set_with_parent)

set(HairRecon_LIBS)
set(HairRecon_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)

# third_party directries
find_package(OpenCV REQUIRED)
set(HairRecon_INCLUDE_DIRS ${HairRecon_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
set(HairRecon_LIBS ${HairRecon_LIBS} ${OpenCV_LIBS})

set(UGU_USE_OPENCV ON CACHE BOOL "" FORCE)
set(UGU_USE_STB OFF CACHE BOOL "" FORCE)
set(UGU_USE_TINYOBJLOADER ON CACHE BOOL "" FORCE)
set(UGU_USE_LODEPNG OFF CACHE BOOL "" FORCE)
set(UGU_USE_TINYCOLORMAP OFF CACHE BOOL "" FORCE)
set(UGU_USE_OPENMP ON CACHE BOOL "" FORCE)
set(UGU_USE_NANORT ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/ugu)
add_definitions(-DUGU_USE_OPENCV)
add_definitions(-DUGU_USE_TINYOBJLOADER)
add_definitions(-DUGU_USE_OPENMP)
add_definitions(-DUGU_USE_NANORT)

set(HairRecon_INCLUDE_DIRS ${HairRecon_INCLUDE_DIRS} ${Ugu_INCLUDE_DIRS})
set(HairRecon_LIBS ${HairRecon_LIBS} ${Ugu_LIBS})

option(HAIRRECON_USE_OPENMP "Use OpenMP to enable parallelization" ON)
message("HAIRRECON_USE_OPENMP: ${HAIRRECON_USE_OPENMP}")
if(HAIRRECON_USE_OPENMP)
  add_definitions(-DHAIRRECON_USE_OPENMP)
endif()

# For OpenMP
if(HAIRRECON_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

set(HairRecon_LIB hairrecon)
add_library(${HairRecon_LIB}
  STATIC
  src/orientation_2d.cpp
  src/lpmvs.cpp
  include/orientation_2d.h
  include/lpmvs.h
  include/util.h
)

set(HairRecon_LIBS ${HairRecon_LIBS} ${HairRecon_LIB})
                            
set_with_parent(HairRecon_LIBS "${HairRecon_LIBS}" "HairRecon_LIBS")
set_with_parent(HairRecon_INCLUDE_DIRS "${HairRecon_INCLUDE_DIRS}" "HairRecon_INCLUDE_DIRS")
message("HairRecon_LIBS: ${HairRecon_LIBS}") 
message("HairRecon_INCLUDE_DIRS: ${HairRecon_INCLUDE_DIRS}") 

target_include_directories(${HairRecon_LIB} PUBLIC ${HairRecon_INCLUDE_DIRS})
set_target_properties(${HairRecon_LIB} PROPERTIES VERSION ${PROJECT_VERSION})



if (NOT hasParent)


  add_executable(orientation_2d_test
    app/orientation_2d_test.cpp)
  target_include_directories(orientation_2d_test PRIVATE ${HairRecon_INCLUDE_DIRS})
  target_link_libraries(orientation_2d_test
    ${HairRecon_LIBS}
    )

  add_executable(gabor_filter_test
    app/gabor_filter_test.cpp)
  target_include_directories(gabor_filter_test PRIVATE ${HairRecon_INCLUDE_DIRS})
  target_link_libraries(gabor_filter_test
    ${HairRecon_LIBS}
    )

  add_executable(render_hair
    app/render_hair.cpp)
  target_include_directories(render_hair PRIVATE ${HairRecon_INCLUDE_DIRS})
  target_link_libraries(render_hair
    ${HairRecon_LIBS}
    )

  add_executable(lpmvs_test
    app/lpmvs_test.cpp)
  target_include_directories(lpmvs_test PRIVATE ${HairRecon_INCLUDE_DIRS} ./third_party/ugu/third_party/json/single_include/)
  target_link_libraries(lpmvs_test
    ${HairRecon_LIBS}
    )
endif()

if (WIN32)
  #set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${EXAMPLES_EXE})
  # suppress C2338 for eigen
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif()



