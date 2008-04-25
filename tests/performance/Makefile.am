#Automake configuration file for hpx/tests

AUTOMAKE_OPTIONS = foreign

bin_PROGRAMS = measure_bind_latest measure_getprefix                     \
               measure_getrange  measure_queryid                         \
               measure_registerid measure_resolve                        \
               measure_unbind  measure_unregister

SUFF = -gcc$(shell $(top_builddir)/gccver)-mt

measure_bind_latest_LDADD = $(top_builddir)/src/libhpx.a                 \
                           -L /home/hkaiser/packages/boost/lib           \
			   -lboost_filesystem$(SUFF)                     \
                           -lboost_date_time$(SUFF)			 \
			   -lboost_thread$(SUFF)                         \
                           -lboost_system$(SUFF)			 \
			   -lboost_serialization$(SUFF)
measure_bind_latest_LDFLAGS = -pthread
measure_bind_latest_SOURCES = measure_bind_latest.cpp

measure_getprefix_LDADD = $(top_builddir)/src/libhpx.a                   \
                	 -L /home/hkaiser/packages/boost/lib             \
			 -lboost_filesystem$(SUFF)                       \
	                 -lboost_date_time$(SUFF) 			 \
			 -lboost_thread$(SUFF) 				 \
			 -lboost_system$(SUFF)				 \
			 -lboost_serialization$(SUFF)
measure_getprefix_LDFLAGS = -pthread
measure_getprefix_SOURCES =measure_getprefix.cpp

measure_getrange_LDADD = $(top_builddir)/src/libhpx.a                    \
	                -L /home/hkaiser/packages/boost/lib              \
			-lboost_filesystem$(SUFF)                        \
		        -lboost_date_time$(SUFF) 		 	 \
			-lboost_thread$(SUFF)           		 \
			-lboost_system$(SUFF)				 \
			-lboost_serialization$(SUFF)
measure_getrange_LDFLAGS = -pthread
measure_getrange_SOURCES =measure_getrange.cpp

measure_queryid_LDADD = $(top_builddir)/src/libhpx.a                     \
 	                -L /home/hkaiser/packages/boost/lib              \
                        -lboost_filesystem$(SUFF)                        \
        	        -lboost_date_time$(SUFF) 			 \
		        -lboost_thread$(SUFF)                            \
		        -lboost_system$(SUFF)				 \
			-lboost_serialization$(SUFF)
measure_queryid_LDFLAGS = -pthread
measure_queryid_SOURCES =measure_queryid.cpp

measure_registerid_LDADD = $(top_builddir)/src/libhpx.a                  \
             	        -L /home/hkaiser/packages/boost/lib              \
		        -lboost_filesystem$(SUFF)                        \
                        -lboost_date_time$(SUFF) 	                 \
	   	        -lboost_thread$(SUFF)                            \
		        -lboost_system$(SUFF)				 \
		        -lboost_serialization$(SUFF)
measure_registerid_LDFLAGS = -pthread
measure_registerid_SOURCES =measure_registerid.cpp

measure_resolve_LDADD = $(top_builddir)/src/libhpx.a                     \
        	        -L /home/hkaiser/packages/boost/lib              \
			-lboost_filesystem$(SUFF)                        \
 	                -lboost_date_time$(SUFF) 			 \
			-lboost_thread$(SUFF)                            \
	        	-lboost_system$(SUFF)				 \
			-lboost_serialization$(SUFF)
measure_resolve_LDFLAGS = -pthread
measure_resolve_SOURCES =measure_resolve.cpp

measure_unbind_LDADD = $(top_builddir)/src/libhpx.a                      \
	                -L /home/hkaiser/packages/boost/lib              \
			-lboost_filesystem$(SUFF)                        \
	                -lboost_date_time$(SUFF) 			 \
			-lboost_thread$(SUFF)                            \
	                -lboost_system$(SUFF)				 \
			-lboost_serialization$(SUFF)
measure_unbind_LDFLAGS = -pthread
measure_unbind_SOURCES =measure_unbind.cpp

measure_unregister_LDADD = $(top_builddir)/src/libhpx.a                  \
	                -L /home/hkaiser/packages/boost/lib              \
			-lboost_filesystem$(SUFF)                        \
	                -lboost_date_time$(SUFF)			 \
		        -lboost_thread$(SUFF)                            \
		        -lboost_system$(SUFF)				 \
			-lboost_serialization$(SUFF)
measure_unregister_LDFLAGS = -pthread
measure_unregister_SOURCES =measure_unregister.cpp

