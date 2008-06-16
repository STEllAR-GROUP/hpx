AUTOMAKE_OPTIONS = foreign
bin_PROGRAMS = accumulator_server accumulator_client
SUFF = -gcc$(shell "gcc -v 2>&1 | grep 'gcc version' | sed 's/^[^345]*//g' | sed 's/\.[0123456789] .*//g' | sed 's/\.//g'")-mt

accumulator_server_LDADD = $(top_builddir)/src/libdistpx.a \
    -L $(HOME)/packages/boost/lib \
    -lboost_date_time$(SUFF) \
    -lboost_thread$(SUFF) \
    -lboost_system$(SUFF) \
    -lboost_serialization$(SUFF)

accumulator_client_LDADD = $(top_builddir)/src/libdistpx.a  \
    -L $(HOME)/packages/boost/lib \
    -lboost_date_time$(SUFF) \
    -lboost_thread$(SUFF) \
    -lboost_system$(SUFF) \
    -lboost_serialization$(SUFF)

accumulator_server_LDFLAGS = -pthread
accumulator_client_LDFLAGS = -pthread
accumulator_server_SOURCES = accumulator_server.cpp
accumulator_client_SOURCES = accumulator_client.cpp



