# Copyright (c) 2007-2013 Hartmut Kaiser
# Copyright (c) 2011-2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
# Automatic Preprocessing. Used to preprocess C++ macro iterations
################################################################################

# add headers to preprocessing
set(function_argument_limit_preprocessing_files
  "hpx/util/detail/function_template.hpp"
  "hpx/util/detail/get_table.hpp"
  "hpx/util/detail/vtable.hpp"
  "hpx/util/detail/vtable_ptr.hpp"
  "hpx/util/detail/vtable_ptr_base.hpp"
  "hpx/util/bind.hpp"
  "hpx/util/bind_action.hpp")
foreach(file ${function_argument_limit_preprocessing_files})
  hpx_partial_preprocess_header(${file} LIMIT HPX_FUNCTION_ARGUMENT_LIMIT)
endforeach()

set(tuple_argument_limit_preprocessing_files
  "hpx/util/tuple.hpp")
foreach(file ${tuple_argument_limit_preprocessing_files})
  hpx_partial_preprocess_header(${file} LIMIT HPX_TUPLE_LIMIT)
endforeach()

set(lock_argument_limit_preprocessing_files
  "hpx/util/locking_helpers.hpp")
foreach(file ${lock_argument_limit_preprocessing_files})
  hpx_partial_preprocess_header(${file} LIMIT HPX_LOCK_LIMIT)
endforeach()

set(action_argument_limit_preprocessing_files
  "hpx/apply.hpp"
  "hpx/async.hpp"
  "hpx/lcos/async.hpp"
  "hpx/lcos/async_fwd.hpp"
  "hpx/lcos/async_continue.hpp"
  "hpx/lcos/async_continue_fwd.hpp"
  "hpx/lcos/packaged_action_constructors.hpp"
  "hpx/lcos/packaged_action_constructors_direct.hpp"
  "hpx/lcos/deferred_packaged_task_constructors.hpp"
  "hpx/lcos/deferred_packaged_task_constructors_direct.hpp"
  "hpx/runtime/actions/component_action_implementations.hpp"
  "hpx/runtime/actions/gcc44/component_action_implementations.hpp"
  "hpx/runtime/actions/component_const_action_implementations.hpp"
  "hpx/runtime/actions/component_non_const_action_implementations.hpp"
  "hpx/runtime/actions/construct_continuation_function_objects.hpp"
  "hpx/runtime/actions/construct_continuation_functions.hpp"
  "hpx/runtime/actions/define_function_operators.hpp"
  "hpx/runtime/actions/plain_action_implementations.hpp"
  "hpx/runtime/actions/transfer_action_constructors.hpp"
  "hpx/runtime/actions/transfer_action_constructors_no_parent_reference.hpp"
  "hpx/runtime/applier/apply_implementations.hpp"
  "hpx/runtime/applier/apply_implementations_callback.hpp"
  "hpx/runtime/applier/apply_continue.hpp"
  "hpx/runtime/components/new.hpp"
  "hpx/runtime/components/server/runtime_support.hpp"
  "hpx/runtime/components/server/create_component_with_args.hpp"
  "hpx/runtime/components/server/runtime_support_create_component_decl.hpp")
foreach(file ${action_argument_limit_preprocessing_files})
  hpx_partial_preprocess_header(${file} LIMIT HPX_ACTION_ARGUMENT_LIMIT)
endforeach()

set(wait_argument_limit_preprocessing_files
  "hpx/lcos/wait_all.hpp"
  "hpx/lcos/wait_n.hpp"
  "hpx/lcos/wait_any.hpp"
  "hpx/lcos/future_wait.hpp"
  "hpx/runtime/components/memory_block.hpp")
foreach(file ${wait_argument_limit_preprocessing_files})
  hpx_partial_preprocess_header(${file} LIMIT HPX_WAIT_ARGUMENT_LIMIT)
endforeach()

# invoke preprocessing routines
hpx_setup_partial_preprocess_headers()


