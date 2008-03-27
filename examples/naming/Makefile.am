# Automake file for hpx/examples/naming

AUTOMAKE_OPTIONS = foreign
bin_PROGRAMS = resolver_server resolver_client
SUFF = -gcc$(shell $(top_builddir)/gccver)-mt

resolver_server_LDADD =                     \
    $(top_builddir)/src/libhpx.a            \
    -L $(HOME)/boost/lib                    \
    -lboost_date_time$(SUFF)                \
    -lboost_thread$(SUFF)                   \
    -lboost_system$(SUFF)                   \
    -lboost_serialization$(SUFF)

resolver_client_LDADD =                     \
    $(top_builddir)/src/libhpx.a            \
    -L $(HOME)/boost/lib                    \
    -lboost_date_time$(SUFF)                \
    -lboost_thread$(SUFF)                   \
    -lboost_system$(SUFF)                   \
    -lboost_serialization$(SUFF)

resolver_server_LDFLAGS = -pthread
resolver_client_LDFLAGS = -pthread

resolver_server_SOURCES = resolver_server.cpp
resolver_client_SOURCES = resolver_client.cpp
