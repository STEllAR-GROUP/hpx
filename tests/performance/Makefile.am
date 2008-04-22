#Automake configuration file for distpx/tests

AUTOMAKE_OPTIONS = foreign

bin_PROGRAMS = measure_bind_latest measure_getprefix measure_queryid  \
measure_registerid measure_resolve measure_unbind  measure_unregister


measure_bind_latest_LDADD = $(top_builddir)/src/libdistpx.a \
               -lboost_date_time -lboost_thread -lboost_asio
measure_bind_latest_LDFLAGS = -pthread
measure_bind_latest_SOURCES = measure_bind_latest.cpp

measure_getprefix_LDADD = $(top_builddir)/src/libdistpx.a \
               -lboost_date_time -lboost_thread -lboost_asio
measure_getprefix_LDFLAGS = -pthread
measure_getprefix_SOURCES =measrue_getprefix.cpp

measure_queryid_LDADD = $(top_builddir)/src/libdistpx.a \
               -lboost_date_time -lboost_thread -lboost_asio
measure_queryid_LDFLAGS = -pthread
measure_queryid_SOURCES =measure_queryid.cpp

measure_registerid_LDADD = $(top_builddir)/src/libdistpx.a \
               -lboost_date_time -lboost_thread -lboost_asio
measure_registerid_LDFLAGS = -pthread
measure_registerid_SOURCES =measure_registerid.cpp

measure_resolve_LDADD = $(top_builddir)/src/libdistpx.a \
               -lboost_date_time -lboost_thread -lboost_asio
measure_resolve_LDFLAGS = -pthread
measure_resolve_SOURCES =measure_resolve.cpp

measure_unbind_LDADD = $(top_builddir)/src/libdistpx.a \
               -lboost_date_time -lboost_thread -lboost_asio
measure_unbind_LDFLAGS = -pthread
measure_unbind_SOURCES =measure_unbind.cpp

measure_unregister_LDADD = $(top_builddir)/src/libdistpx.a \
               -lboost_date_time -lboost_thread -lboost_asio
measure_unregister_LDFLAGS = -pthread
measure_unregister_SOURCES =measure_unregister.cpp

