# Automake file for hpx/examples/naming

AUTOMAKE_OPTIONS = foreign
bin_PROGRAMS = server client
SUFF = -gcc$(shell $(top_builddir)/gccver)-mt

server_LDADD =                              \
    $(top_builddir)/src/libhpx.a            \
    -L /home/hkaiser/packages/boost/lib/    \
    -lboost_filesystem$(SUFF)               \
    -lboost_date_time$(SUFF)                \
    -lboost_thread$(SUFF)                   \
    -lboost_system$(SUFF)                   \
    -lboost_serialization$(SUFF)

client_LDADD =                              \
    $(top_builddir)/src/libhpx.a            \
    -L/home/hkaiser/packages/boost/lib/     \
    -lboost_filesystem$(SUFF)               \
    -lboost_date_time$(SUFF)                \
    -lboost_thread$(SUFF)                   \
    -lboost_system$(SUFF)                   \
    -lboost_serialization$(SUFF)

server_LDFLAGS = -pthread
client_LDFLAGS = -pthread

server_SOURCES = resolver_server.cpp
client_SOURCES = resolver_client.cpp
