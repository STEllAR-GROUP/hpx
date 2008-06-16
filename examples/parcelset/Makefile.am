# Automake file for distpx/examples/parcelset

AUTOMAKE_OPTIONS = foreign
bin_PROGRAMS = parcelset_server parcelset_client
SUFF = -gcc$(shell $(top_builddir)/gccver)-mt

parcelset_server_LDADD =                        \
    $(top_builddir)/src/libhpx.a                \
    -L $(HOME)/packages/boost/lib                        \
    -lboost_date_time$(SUFF)                    \
    -lboost_thread$(SUFF)                       \
    -lboost_system$(SUFF)                       \
    -lboost_serialization$(SUFF)

parcelset_client_LDADD =                        \
    $(top_builddir)/src/libhpx.a                \
    -L $(HOME)/packages/boost/lib                        \
    -lboost_date_time$(SUFF)                    \
    -lboost_thread$(SUFF)                       \
    -lboost_system$(SUFF)                       \
    -lboost_serialization$(SUFF)

parcelset_server_LDFLAGS = -pthread
parcelset_client_LDFLAGS = -pthread

parcelset_server_SOURCES = parcelset_server.cpp
parcelset_client_SOURCES = parcelset_client.cpp
