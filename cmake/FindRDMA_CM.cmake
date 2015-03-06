# - Try to find RDMA CM
# Once done this will define
#  RDMA_CM_FOUND - System has RDMA CM
#  RDMA_CM_INCLUDE_DIRS - The RDMA CM include directories
#  RDMA_CM_LIBRARIES - The libraries needed to use RDMA CM

find_path(RDMA_CM_INCLUDE_DIR rdma_cma.h
  HINTS /usr/local/include /usr/include/rdma)

find_library(RDMA_CM_LIBRARY NAMES rdmacm
  PATHS /usr/local/lib /usr/lib)

set(RDMA_CM_INCLUDE_DIRS ${RDMA_CM_INCLUDE_DIR})
set(RDMA_CM_LIBRARIES ${RDMA_CM_LIBRARY})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set RDMA_CM_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(RDMA_CM DEFAULT_MSG
                                  RDMA_CM_INCLUDE_DIR RDMA_CM_LIBRARY)

mark_as_advanced(RDMA_CM_INCLUDE_DIR RDMA_CM_LIBRARY)
