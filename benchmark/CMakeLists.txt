cmake_minimum_required(VERSION 3.0)

project(pytorch_motion_planner_benchmark)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(ompl 1.5 REQUIRED)
include_directories(${OMPL_INCLUDE_DIR})

find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(PkgConfig REQUIRED)
pkg_check_modules(SBPL REQUIRED sbpl)
include_directories(${SBPL_INCLUDE_DIRS})
link_directories(${SBPL_LIBRARY_DIRS})

find_package(Boost 1.58 REQUIRED COMPONENTS serialization filesystem system program_options)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS bench_mr_binding/*.h)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS bench_mr_binding/*.cpp)

set(BENCHMARK_DIR /home/mikhail/research/bench-mr)
include_directories(${BENCHMARK_DIR}/src)
include_directories(${BENCHMARK_DIR}/third_party/chomp)
include_directories(${BENCHMARK_DIR}/third_party/params/include)

link_directories(${BENCHMARK_DIR}/build/)
link_directories(${BENCHMARK_DIR}/build/lib)
link_directories(${BENCHMARK_DIR}/build/third_party/steering_functions)

set(EXTRA_LIB ${EXTRA_LIB} ${SBPL_LIBRARIES} ${Boost_LIBRARIES} ${OMPL_LIBRARIES} chomp mzcommon steering_functions)
set(EXTRA_LIB ${EXTRA_LIB} mpb_core mpb_steering mpb_planners mpb_smoothers)

add_executable(test ${SOURCES})
target_link_libraries(test ${EXTRA_LIB} ${EXTRA_LIB})
