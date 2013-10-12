//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_registry.hpp>
#include <hpx/runtime/components/base_lco_factory.hpp>
#include <hpx/lcos/promise.hpp>

#include <string>

HPX_REGISTER_PROMISE(hpx::lcos::promise<boost::int16_t>, int16_t_promise,
    hpx::components::int16_t_promise)
HPX_REGISTER_PROMISE(hpx::lcos::promise<boost::uint16_t>, uint16_t_promise,
    hpx::components::uint16_t_promise)
HPX_REGISTER_PROMISE(hpx::lcos::promise<boost::int32_t>, int32_t_promise,
    hpx::components::int32_t_promise)
HPX_REGISTER_PROMISE(hpx::lcos::promise<boost::uint32_t>, uint32_t_promise,
    hpx::components::uint32_t_promise)
HPX_REGISTER_PROMISE(hpx::lcos::promise<std::string>, string_promise,
    hpx::components::string_promise)
