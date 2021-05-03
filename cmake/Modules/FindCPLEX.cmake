# This module finds cplex.
#
# User can give CPLEX_ROOT_DIR as a hint stored in the cmake cache.
#
# It sets the following variables:
#  CPLEX_FOUND              - Set to false, or undefined, if cplex isn't found.
#  CPLEX_INCLUDE_DIRS       - include directory
#  CPLEX_LIBRARIES          - library files

if(WIN32)
  execute_process(COMMAND cmd /C set CPLEX_STUDIO_DIR OUTPUT_VARIABLE CPLEX_STUDIO_DIR_VAR ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT CPLEX_STUDIO_DIR_VAR)
    MESSAGE(FATAL_ERROR "Unable to find CPLEX: environment variable CPLEX_STUDIO_DIR<VERSION> not set.")
  endif()

  STRING(REGEX REPLACE "^CPLEX_STUDIO_DIR" "" CPLEX_STUDIO_DIR_VAR ${CPLEX_STUDIO_DIR_VAR})
  STRING(REGEX MATCH "^[0-9]+" CPLEX_WIN_VERSION ${CPLEX_STUDIO_DIR_VAR})
  STRING(REGEX REPLACE "^[0-9]+=" "" CPLEX_STUDIO_DIR_VAR ${CPLEX_STUDIO_DIR_VAR})
  file(TO_CMAKE_PATH "${CPLEX_STUDIO_DIR_VAR}" CPLEX_ROOT_DIR_GUESS)

  set(CPLEX_WIN_VERSION ${CPLEX_WIN_VERSION} CACHE STRING "CPLEX version to be used.")
  set(CPLEX_ROOT_DIR "${CPLEX_ROOT_DIR_GUESS}" CACHE PATH "CPLEX root directory.")

  MESSAGE(STATUS "Found CLPEX version ${CPLEX_WIN_VERSION} at '${CPLEX_ROOT_DIR}'")

  STRING(REGEX REPLACE "/VC/bin/.*" "" VISUAL_STUDIO_PATH ${CMAKE_C_COMPILER})
  STRING(REGEX MATCH "Studio [0-9]+" CPLEX_WIN_VS_VERSION ${VISUAL_STUDIO_PATH})
  STRING(REGEX REPLACE "Studio " "" CPLEX_WIN_VS_VERSION ${CPLEX_WIN_VS_VERSION})

  if(${CPLEX_WIN_VS_VERSION} STREQUAL "9")
    set(CPLEX_WIN_VS_VERSION 2008)
  elseif(${CPLEX_WIN_VS_VERSION} STREQUAL "10")
    set(CPLEX_WIN_VS_VERSION 2010)
  elseif(${CPLEX_WIN_VS_VERSION} STREQUAL "11")
    set(CPLEX_WIN_VS_VERSION 2012)
  else()
    MESSAGE(FATAL_ERROR "CPLEX: unknown Visual Studio version at '${VISUAL_STUDIO_PATH}'.")
  endif()

  set(CPLEX_WIN_VS_VERSION ${CPLEX_WIN_VS_VERSION} CACHE STRING "Visual Studio Version")

  if("${CMAKE_C_COMPILER}" MATCHES "amd64")
    set(CPLEX_WIN_BITNESS x64)
  else()
    set(CPLEX_WIN_BITNESS x86)
  endif()

  set(CPLEX_WIN_BITNESS ${CPLEX_WIN_BITNESS} CACHE STRING "On Windows: x86 or x64 (32bit resp. 64bit)")

  MESSAGE(STATUS "CPLEX: using Visual Studio ${CPLEX_WIN_VS_VERSION} ${CPLEX_WIN_BITNESS} at '${VISUAL_STUDIO_PATH}'")

  if(NOT CPLEX_WIN_LINKAGE)
    set(CPLEX_WIN_LINKAGE mda CACHE STRING "CPLEX linkage variant on Windows. One of these: mda (dll, release), mdd (dll, debug), mta (static, release), mtd (static, debug)")
  endif(NOT CPLEX_WIN_LINKAGE)

  # now, generate platform string
  set(CPLEX_WIN_PLATFORM "${CPLEX_WIN_BITNESS}_windows_vs${CPLEX_WIN_VS_VERSION}/stat_${CPLEX_WIN_LINKAGE}")

else()
  set(CPLEX_ILOG_DIRS /opt/ibm/ILOG /opt/IBM/ILOG)
  if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CPLEX_ARCH x86-64)
  else ()
    set(CPLEX_ARCH x86)
  endif ()
  if (APPLE)
    set(CPLEX_ILOG_DIRS $ENV{HOME}/Applications/IBM/ILOG ${CPLEX_ILOG_DIRS})
    foreach (suffix "osx" "darwin9_gcc4.0")
      set(CPLEX_LIB_PATH_SUFFIXES
          ${CPLEX_LIB_PATH_SUFFIXES} lib/${CPLEX_ARCH}_${suffix}/static_pic)
    endforeach ()
  else ()
    set(CPLEX_LIB_PATH_SUFFIXES
      lib/${CPLEX_ARCH}_sles10_4.1/static_pic lib/${CPLEX_ARCH}_linux/static_pic)
  endif ()
  if (NOT CPLEX_STUDIO_DIR)
    if (CPLEX_ROOT_DIR)
      set(CPLEX_STUDIO_DIR ${CPLEX_ROOT_DIR})
    else ()
      foreach (dir ${CPLEX_ILOG_DIRS})
        file(GLOB CPLEX_STUDIO_DIRS "${dir}/CPLEX_Studio*")
        list(SORT CPLEX_STUDIO_DIRS)
        list(REVERSE CPLEX_STUDIO_DIRS)
        message("Found studio dirs: ${CPLEX_STUDIO_DIRS}")
        if (CPLEX_STUDIO_DIRS)
          list(GET CPLEX_STUDIO_DIRS 0 CPLEX_STUDIO_DIR_)
          string(REGEX MATCH "[0-9][0-9][0-9][0-9]" CPXVERSION ${CPLEX_STUDIO_DIR_})
          message(STATUS "Found CPLEX Studio: ${CPLEX_STUDIO_DIR_}")
          message(STATUS "Detected CPLEX version ${CPXVERSION}")
          break ()
        endif ()
      endforeach ()
      if (NOT CPLEX_STUDIO_DIR_)
        set(CPLEX_STUDIO_DIR_ CPLEX_STUDIO_DIR-NOTFOUND)
      endif ()
      set(CPLEX_STUDIO_DIR ${CPLEX_STUDIO_DIR_} CACHE PATH
        "Path to the CPLEX Studio directory")
      set(CPLEX_ROOT_DIR "${CPLEX_STUDIO_DIR}" CACHE PATH "CPLEX root directory.")
    endif ()
  elseif (NOT CPLEX_ROOT_DIR)
    set(CPLEX_ROOT_DIR "${CPLEX_STUDIO_DIR}" CACHE PATH "CPLEX root directory.")
  endif ()
  set(CPLEX_WIN_PLATFORM "")

endif()


FIND_PATH(CPLEX_INCLUDE_DIR
  ilcplex/cplex.h
  HINTS ${CPLEX_ROOT_DIR}/cplex/include
        ${CPLEX_ROOT_DIR}/include
  PATHS ENV C_INCLUDE_PATH
        ENV C_PLUS_INCLUDE_PATH
        ENV INCLUDE_PATH
  )
message(STATUS "CPLEX include dir: ${CPLEX_INCLUDE_DIR}")

FIND_PATH(CPLEX_CONCERT_INCLUDE_DIR
  ilconcert/iloenv.h
  HINTS ${CPLEX_ROOT_DIR}/concert/include
        ${CPLEX_ROOT_DIR}/include
  PATHS ENV C_INCLUDE_PATH
        ENV C_PLUS_INCLUDE_PATH
        ENV INCLUDE_PATH
  )
message(STATUS "CPLEX CONCERT include dir: ${CPLEX_CONCERT_INCLUDE_DIR}")

FIND_LIBRARY(CPLEX_LIBRARY
  NAMES cplex${CPLEX_WIN_VERSION} cplex
  HINTS ${CPLEX_ROOT_DIR}/cplex/lib/${CPLEX_WIN_PLATFORM} #windows
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_debian4.0_4.1/static_pic #unix
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_sles10_4.1/static_pic #unix
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_pic #unix
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_osx/static_pic #osx
  PATHS ENV LIBRARY_PATH #unix
        ENV LD_LIBRARY_PATH #unix
  )
message(STATUS "CPLEX Library: ${CPLEX_LIBRARY}")

FIND_LIBRARY(CPLEX_ILOCPLEX_LIBRARY
  ilocplex
  HINTS ${CPLEX_ROOT_DIR}/cplex/lib/${CPLEX_WIN_PLATFORM} #windows
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_debian4.0_4.1/static_pic #unix
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_sles10_4.1/static_pic #unix
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_pic #unix
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_osx/static_pic #osx
  PATHS ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
  )
message(STATUS "ILOCPLEX Library: ${CPLEX_ILOCPLEX_LIBRARY}")

FIND_LIBRARY(CPLEX_CONCERT_LIBRARY
  concert
  HINTS ${CPLEX_ROOT_DIR}/concert/lib/${CPLEX_WIN_PLATFORM} #windows
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_debian4.0_4.1/static_pic #unix
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_sles10_4.1/static_pic #unix
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_linux/static_pic #unix
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_osx/static_pic #osx
  PATHS ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
  )
message(STATUS "CONCERT Library: ${CPLEX_CONCERT_LIBRARY}")

if(WIN32)
	FIND_PATH(CPLEX_BIN_DIR
	  cplex${CPLEX_WIN_VERSION}.dll
          HINTS ${CPLEX_ROOT_DIR}/cplex/bin/${CPLEX_WIN_PLATFORM} #windows
	  )
else()
	FIND_PATH(CPLEX_BIN_DIR
	  cplex
          HINTS ${CPLEX_ROOT_DIR}/cplex/bin/x86-64_sles10_4.1 #unix
                ${CPLEX_ROOT_DIR}/cplex/bin/x86-64_debian4.0_4.1 #unix
                ${CPLEX_ROOT_DIR}/cplex/bin/x86-64_linux #unix
                ${CPLEX_ROOT_DIR}/cplex/bin/x86-64_osx #osx
	  ENV LIBRARY_PATH
          ENV LD_LIBRARY_PATH
	  )
endif()
message(STATUS "CPLEX Bin Dir: ${CPLEX_BIN_DIR}")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CPLEX DEFAULT_MSG
 CPLEX_LIBRARY CPLEX_INCLUDE_DIR CPLEX_ILOCPLEX_LIBRARY CPLEX_CONCERT_LIBRARY CPLEX_CONCERT_INCLUDE_DIR)

IF(CPLEX_FOUND)
  SET(CPLEX_INCLUDE_DIRS ${CPLEX_INCLUDE_DIR} ${CPLEX_CONCERT_INCLUDE_DIR})
  SET(CPLEX_LIBRARIES ${CPLEX_CONCERT_LIBRARY} ${CPLEX_ILOCPLEX_LIBRARY} ${CPLEX_LIBRARY} )
  IF(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    SET(CPLEX_LIBRARIES "${CPLEX_LIBRARIES};m;pthread")
  ENDIF(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  SET(CPLEX_LIBRARIES "${CPLEX_LIBRARIES};m;pthread;dl")
ENDIF(CPLEX_FOUND)

MARK_AS_ADVANCED(CPLEX_LIBRARY CPLEX_INCLUDE_DIR CPLEX_ILOCPLEX_LIBRARY CPLEX_CONCERT_INCLUDE_DIR CPLEX_CONCERT_LIBRARY)


