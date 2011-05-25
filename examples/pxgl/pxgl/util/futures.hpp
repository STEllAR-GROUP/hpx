// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_FUTURE(name, suffix)                                      \
HPX_REGISTER_ACTION_EX(                                                        \
  hpx::lcos::base_lco_with_value<name>::set_result_action,                     \
    set_result_action_suffix_type);                                            \
HPX_REGISTER_ACTION_EX(                                                        \
  hpx::lcos::base_lco_with_value<name>::get_value_action,                      \
    get_value_action_suffix_type);                                             \
HPX_DEFINE_GET_COMPONENT_TYPE(                                                 \
  hpx::lcos::base_lco_with_value<name>);

