# - Try to find CBC
# Once done this will define
#  CBC_FOUND           - System has CBC
#  CBC_INCLUDE_DIRS    - The CBC include directories
#  CBC_LIBRARY_PATHS   - The libraries needed to use CBC
#  CBC_TARGETS         - The names of imported targets created for CBC
# User can set CBC_ROOT to the preferred installation prefix

set(COIN_SUB_DIRS "coin-or" "coin")
set(CBC_FIND_FILES CbcModel.hpp ClpSimplex.hpp OsiClpSolverInterface.hpp OsiSolverInterface.hpp CoinPragma.hpp)

foreach(COIN_SUB_DIR ${COIN_SUB_DIRS})
  foreach(CBC_FILE ${CBC_FIND_FILES})
    set(CBC_FILE_LOC "CBC_LIB_LOC-NOTFOUND")
    #message(STATUS "CBC: Looking for file `${COIN_SUB_DIR}/${CBC_FILE}`")
    find_path(CBC_FILE_LOC ${COIN_SUB_DIR}/${CBC_FILE}
              PATH_SUFFIXES cbc cgl clp coinutils osi include)
    if("${CBC_FILE_LOC}" STREQUAL "CBC_FILE_LOC-NOTFOUND")
      #message(STATUS "CBC: Could not find file `${CBC_FILE}`")
      set(CBC_INCLUDE_DIRS "")
      break()
    else()
      #message(STATUS "CBC: Found file `${CBC_FILE}` at `${CBC_FILE_LOC}`")
    endif()
    list(APPEND CBC_INCLUDE_DIRS ${CBC_FILE_LOC})
    list(APPEND CBC_INCLUDE_DIRS ${CBC_FILE_LOC}/${COIN_SUB_DIR})
  endforeach(CBC_FILE)
  if(NOT "${CBC_FILE_LOC}" STREQUAL "CBC_FILE_LOC-NOTFOUND")
    #message(STATUS "CBC: Found CBC_INCLUDE_DIRS `${CBC_INCLUDE_DIRS}`")
    break()
  endif()
endforeach(COIN_SUB_DIR)

if(NOT "${CBC_FILE_LOC}" STREQUAL "CBC_FILE_LOC-NOTFOUND")
  list(REMOVE_DUPLICATES CBC_INCLUDE_DIRS)
  unset(CBC_FIND_FILES)
  unset(CBC_FILE_LOC)

  if(WIN32 AND NOT UNIX)
    set(CBC_REQ_LIBS Osi OsiClp OsiCbc Clp Cgl Cbc CbcSolver CoinUtils)
  else()
    set(CBC_REQ_LIBS CbcSolver Cbc Cgl OsiClp Clp Osi CoinUtils)
  endif()

  foreach(CBC_LIB ${CBC_REQ_LIBS})
    set(CBC_LIB_LOC "CBC_LIB_LOC-NOTFOUND")
    find_library(CBC_LIB_LOC NAMES ${CBC_LIB} lib${CBC_LIB}
                PATH_SUFFIXES lib)
    if("${CBC_LIB_LOC}" STREQUAL "CBC_LIB_LOC-NOTFOUND")
      #message(STATUS "CBC: Could not find library `${CBC_LIB}`")
      set(CBC_LIBRARY_PATHS "")
      break()
    endif()
    list(APPEND CBC_LIBRARY_PATHS ${CBC_LIB_LOC})
    add_library(${CBC_LIB} UNKNOWN IMPORTED)
    set_target_properties(${CBC_LIB} PROPERTIES
                          IMPORTED_LOCATION ${CBC_LIB_LOC}
                          INTERFACE_INCLUDE_DIRECTORIES "${CBC_INCLUDE_DIRS}")
    list(APPEND CBC_TARGETS ${CBC_LIB})
  endforeach(CBC_LIB)

  unset(CBC_REQ_LIBS)
  unset(CBC_LIB_LOC)

  if(UNIX AND NOT WIN32 AND NOT DEFINED EMSCRIPTEN)
    find_package(ZLIB)
    if(NOT ZLIB_FOUND)
      message(STATUS "CBC: Missing dependency `Zlib`")
      set(CBC_LIBRARY_PATHS "")
    else()
      list(APPEND CBC_LIBRARY_PATHS ${ZLIB_LIBRARIES})
      list(APPEND CBC_TARGETS ${ZLIB_LIBRARIES})
    endif()
  endif()
endif()

#message(STATUS "CBC: CBC_INCLUDE_DIRS = `${CBC_INCLUDE_DIRS}`")
#message(STATUS "CBC: CBC_LIBRARY_PATHS = `${CBC_LIBRARY_PATHS}`")
#message(STATUS "CBC: CBC_TARGETS = `${CBC_TARGETS}`")

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set CBC_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(CBC
  FOUND_VAR CBC_FOUND
  REQUIRED_VARS CBC_INCLUDE_DIRS CBC_LIBRARY_PATHS
  FAIL_MESSAGE "Could NOT find CBC, use CBC_ROOT to hint its location"
)

mark_as_advanced(CBC_INCLUDE_DIRS CBC_LIBRARY_PATHS)
