# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Set examples to build by default
set(example_SUBDIRS 
	#quickstart
	bh
	##had_amr
   )

# Pick up additional examples for this build set using
# -DHPX_WITH_EXAMPLES="ex1;ex2;ex3"
if(HPX_WITH_EXAMPLES)
  LIST(APPEND example_SUBDIRS ${HPX_WITH_EXAMPLES})
endif(HPX_WITH_EXAMPLES)
LIST(REMOVE_DUPLICATES example_SUBDIRS)

message("Building HPX with examples:")
foreach(subdir ${example_SUBDIRS})
message("  " ${subdir})
    add_subdirectory(${subdir})
endforeach()

