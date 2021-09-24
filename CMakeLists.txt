# Copyright 2021
# Alexander Belyi <alexander.belyi@gmail.com>
# 
# This file is part of BestPartition project.
# 
# BestPartition is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# BestPartition is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with BestPartition. If not, see <http://www.gnu.org/licenses/>.
 
 cmake_minimum_required(VERSION 3.10)

# set the project name
project(BestPartition VERSION 0.1)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif(POLICY CMP0074)

# specify target executable
add_executable(BestPartition Main.cpp Combo/Combo.cpp Combo/Graph.cpp BestPartition.cpp BranchAndBound.cpp PenalizingSubnetworks.cpp ILP.cpp)

#includes path
target_include_directories(BestPartition PRIVATE "${CMAKE_SOURCE_DIR}/Combo/")

#find CBC
find_package(CBC)
#[[
if (NOT CBC_FOUND)
    #add CBC libs manually
    #this code is here to add debug version of CBC
    #most likely you don't want it to be executed
    #if you have errors finding CBC make sure to install it correctly and/or provide CBC_ROOT
    if (NOT CBC_ROOT)
        set(CBC_ROOT ${CMAKE_SOURCE_DIR}/../Cbc-2.9/)
    endif ()
    set(CBC_LIBS_DIR_DBG ${CBC_ROOT}/build/debug/lib)
    set(CBC_LIBS_DIR_REL ${CBC_ROOT}/build/release/lib)

    set(CBC_TARGETS Cbc OsiClp Clp CoinUtils)
    foreach(lib IN LISTS CBC_TARGETS)
        add_library(${lib} UNKNOWN IMPORTED)
        set_target_properties(${lib}
            PROPERTIES
                IMPORTED_LOCATION_DEBUG   ${CBC_LIBS_DIR_DBG}/lib${lib}.a
                IMPORTED_LOCATION_RELEASE ${CBC_LIBS_DIR_REL}/lib${lib}.so
            )
    endforeach()
    #Degug-only static libs
    set(CBC_LIBS_DBG Cgl Osi coinmumps coinglpk coinlapack coinmetis coinblas)
    foreach(lib IN LISTS CBC_LIBS_DBG)
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib}
            PROPERTIES
                IMPORTED_LOCATION_DEBUG   ${CBC_LIBS_DIR_DBG}/lib${lib}.a
            )
    endforeach()
    set(CBC_INCLUDE_DIRS ${CBC_ROOT}/build/release/include/coin/)
endif ()
#]]

#includes path
target_include_directories(BestPartition PRIVATE ${CBC_INCLUDE_DIRS})

#link libraries
target_link_libraries(BestPartition ${CBC_TARGETS})

#check if CPLEX installed
find_package(CPLEX)
if (CPLEX_FOUND)
    target_include_directories(BestPartition PRIVATE ${CPLEX_INCLUDE_DIRS})
    #add CPLEX libraries
    target_link_libraries(BestPartition ${CPLEX_LIBRARIES} dl)
    #[[
     #this code is for linking CPLEX dynamically
     string(REGEX MATCH "[0-9][0-9][0-9][0-9]" CPXVERSION ${CPLEX_STUDIO_DIR})
     set(CPLEX_DYN_LIB "${CPLEX_BIN_DIR}/libcplex${CPXVERSION}0.so")
     target_link_libraries(BestPartition ${CPLEX_DYN_LIB} ${CPLEX_ILOCPLEX_LIBRARY} ${CPLEX_CONCERT_LIBRARY} m pthread)
    #]]
    #set flag that CPLEX exists
    add_compile_definitions(IL_STD CPLEX_AVAILABLE)
endif()

#[[
if (NOT CBC_FOUND)
    #debug -> link the rest of CBC static libs
    #this should be added after CPLEX since coinblas conflicts with libcplex
    foreach(lib IN LISTS CBC_LIBS_DBG)
        target_link_libraries(BestPartition debug ${lib})
    endforeach()
    target_link_libraries(BestPartition debug gfortran debug z debug dl)
endif ()
#]]
