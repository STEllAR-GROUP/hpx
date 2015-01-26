# Copyright (c) 2007-2014 Hartmut Kaiser
# Copyright (c) 2011-2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
# Automatic Preprocessing. Used to preprocess C++ macro iterations
################################################################################

# add headers to preprocessing
set(tuple_argument_limit_preprocessing_files
  "hpx/util/tuple.hpp")
foreach(file ${tuple_argument_limit_preprocessing_files})
  hpx_partial_preprocess_header(${file} LIMIT HPX_LIMIT)
endforeach()

set(action_argument_limit_preprocessing_files
  "hpx/lcos/broadcast.hpp"
  "hpx/lcos/reduce.hpp")
foreach(file ${action_argument_limit_preprocessing_files})
  hpx_partial_preprocess_header(${file} LIMIT HPX_LIMIT)
endforeach()

set(wait_argument_limit_preprocessing_files
  "hpx/runtime/components/memory_block.hpp")
foreach(file ${wait_argument_limit_preprocessing_files})
  hpx_partial_preprocess_header(${file} LIMIT HPX_LIMIT)
endforeach()

# invoke preprocessing routines
hpx_setup_partial_preprocess_headers()


