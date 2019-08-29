# Copyright (c) 2019 Ste||ar Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# FIXME : in the future put it directly inside the cmake directory of the
# corresponding plugin

# FIXME : not sure if this comment is still up-to-date
# If we compile with the MPI parcelport enabled, we need to additionally
# add the MPI include path here, because for the main library, it's only
# added for the plugin.
if(HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_MPI)
  find_package(MPI)
  # All cmake version don't have the same found variable set
  if(NOT MPI_FOUND AND NOT MPI_CXX_FOUND)
    hpx_error("MPI could not be found and HPX_WITH_PARCELPORT_MPI=ON, please specify \
    MPI_ROOT to point to the root of your MPI installation")
  endif()
  add_library(hpx::mpi INTERFACE IMPORTED)
  set_property(TARGET hpx::mpi PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${MPI_INCLUDE_PATH} ${MPI_CXX_INCLUDE_DIRS})
  # MPI_LIBRARY and EXTRA is deprecated but still linked for older MPI versions
  if (MPI_CXX_LIBRARIES)
    set_property(TARGET hpx::mpi APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES ${MPI_CXX_LIBRARIES})
  endif()
  # Insure compatibility with older versions
  if (MPI_LIBRARY)
    set_property(TARGET hpx::mpi APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES ${MPI_LIBRARY})
  endif()
  if (MPI_EXTRA_LIBRARY)
    set_property(TARGET hpx::mpi APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES ${MPI_EXTRA_LIBRARY})
  endif()
  set_property(TARGET hpx::mpi PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
  set_property(TARGET hpx::mpi PROPERTY INTERFACE_COMPILE_DEFINITIONS ${MPI_CXX_COMPILE_DEFINITIONS})
  if(MPI_CXX_LINK_FLAGS)
    #hpx_add_link_flag_if_available(${MPI_CXX_LINK_FLAGS})
  endif()
endif()
