# Autoconf configuration file for distpx
# Todo: Checking for needed libraries with autoconf.

AC_INIT(src/core/px.cpp)
AC_CONFIG_AUX_DIR(config)
AM_INIT_AUTOMAKE(libhpx.a,0.1)
AC_PROG_CC
AC_PROG_CXX
AC_PROG_RANLIB

CXXFLAGS = '-I$(top_builddir) -I$(HOME)/boost/include/boost-1_34_1/ -I$(top_builddir)/external/endian -I$(top_builddir)/external/singleton -I$(top_builddir)/external/coroutine -I$(top_builddir)/external/endian -I$(top_builddir)/external/function_types -g -Wall'

AC_OUTPUT(Makefile                           \
#          tests/Makefile                     \
#          tests/performance/Makefile         \
#          examples/accumulator/Makefile      \
          examples/naming/Makefile           \
          examples/parcelset/Makefile        \
#          examples/graph_component/Makefile  \
          src/Makefile)
