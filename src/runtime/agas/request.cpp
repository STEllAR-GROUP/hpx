////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>

HPX_UTIL_REGISTER_FUNCTION(
    void(std::string const&, hpx::naming::gid_type const&)
  , hpx::util::function<void(std::string const&, hpx::naming::gid_type const&)>
  , request_iterate_names_function_type
)

HPX_UTIL_REGISTER_FUNCTION(
    void(std::string const&, hpx::components::component_type)
  , hpx::util::function<void(std::string const&, hpx::components::component_type)>
  , request_iterate_types_function_type
)

