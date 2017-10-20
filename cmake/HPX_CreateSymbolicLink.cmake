# Copyright (c) 2017 Denis Blank
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Creates a symbolic link from the destination to the target,
# if the link doesn't exist yet.
# Since `create_symlink` is only available for unix derivates,
# we work around that in this macro.
macro(create_symbolic_link SYM_TARGET SYM_DESTINATION)
  if(WIN32)
    if(NOT EXISTS ${SYM_DESTINATION})
      # Create a junction for windows links
      execute_process(COMMAND cmd /C "${CMAKE_SOURCE_DIR}/cmake/scripts/create_symbolic_link.bat"
                                     ${SYM_DESTINATION} ${SYM_TARGET})
    endif()
  else()
    # Only available on unix derivates
    execute_process(COMMAND "${CMAKE_COMMAND}" -E create_symlink ${SYM_TARGET} ${SYM_DESTINATION})
  endif()
endmacro(create_symbolic_link)
