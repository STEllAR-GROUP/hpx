# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_program(FOP_EXECUTABLE
  NAMES
    fop
    xep.bat
  PATHS
    $ENV{PROGRAMFILES}/RenderX/XEP
    ${FOP_ROOT}
    ENV FOP_ROOT
  DOC
    "An XSL-FO processor"
  )

if(FOP_EXECUTABLE)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(FOP DEFAULT_MSG FOP_EXECUTABLE)
endif()
