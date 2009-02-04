//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "mandelbrot_callback.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace mandelbrot { namespace server { namespace detail
{
    void callback::set_result (mandelbrot::result const& result)
    {
        if (f_) f_(result);
    }

}}}

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE();
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    mandelbrot::server::callback, mandelbrot_callback);
HPX_DEFINE_GET_COMPONENT_TYPE(mandelbrot::server::detail::callback);

HPX_REGISTER_ACTION_EX(
    hpx::lcos::base_lco_with_value<mandelbrot::result>::set_result_action,
    set_result_action_mandelbrot_result);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<mandelbrot::result>);

