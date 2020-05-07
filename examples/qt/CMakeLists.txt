# Copyright (c) 2012-2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_EXAMPLES_QT4)
  include(${QT_USE_FILE})

  qt4_wrap_cpp(
    QT_MOC_SOURCES widget.hpp OPTIONS -DBOOST_TT_HAS_OPERATOR_HPP_INCLUDED
  )

  set(qt_sources
      qt.cpp widget.cpp
      # hpx_qt.cpp
      ${QT_MOC_SOURCES}
  )

  add_hpx_executable(
    qt INTERNAL_FLAGS MODULE qt
    SOURCES ${qt_sources} ${qt_FLAGS}
    FOLDER "Examples/Qt"
  )

  target_include_directories(
    qt
    PRIVATE ./
    PRIVATE ${QT4_INCLUDE_DIRS}
  )
  target_link_libraries(qt PRIVATE ${HPX_TLL_PUBLIC} ${QT_LIBRARIES})

  add_hpx_example_target_dependencies("qt" qt)

endif()
