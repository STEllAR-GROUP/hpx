////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <string>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/lcos/base_lco.hpp>

using hpx::lcos::base_lco_with_value;

HPX_REGISTER_ACTION_EX
    (base_lco_with_value<bool>::set_result_action,
     set_result_action_int);

HPX_REGISTER_ACTION_EX
    (base_lco_with_value<std::string>::set_result_action,
     set_result_action_string);

HPX_REGISTER_ACTION_EX
    (base_lco_with_value<std::vector<boost::uint32_t> >::set_result_action,
     set_result_action_prefix_vector);

