//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_registry.hpp>
#include <hpx/runtime/components/base_lco_factory.hpp>
#include <hpx/lcos/promise.hpp>

HPX_REGISTER_PROMISE(hpx::lcos::promise<void>, gid_promise)
HPX_REGISTER_PROMISE(hpx::lcos::promise<float>, float_promise)
HPX_REGISTER_PROMISE(hpx::lcos::promise<double>, double_promise)
HPX_REGISTER_PROMISE(hpx::lcos::promise<boost::int8_t>, int8_t_promise)
HPX_REGISTER_PROMISE(hpx::lcos::promise<boost::uint8_t>, uint8_t_promise)
