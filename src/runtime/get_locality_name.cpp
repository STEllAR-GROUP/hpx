//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/async.hpp>
#include <hpx/apply.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/get_locality_name.hpp>
#include <hpx/runtime.hpp>

#include <string>

#include <boost/lexical_cast.hpp>

namespace hpx { namespace detail
{
    std::string get_locality_base_name()
    {
        runtime* rt = get_runtime_ptr();
        if (rt == 0)
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::detail::get_locality_name",
                "the runtime system is not operational at this point");
            return "";
        }
        return rt->get_parcel_handler().get_locality_name();
    }

    std::string get_locality_name()
    {
        std::string basename = get_locality_base_name();
        return basename + '#' + boost::lexical_cast<std::string>(get_locality_id());
    }
}}

HPX_PLAIN_ACTION(hpx::detail::get_locality_name, hpx_get_locality_name_action);

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    std::string get_locality_name()
    {
        return detail::get_locality_name();
    }

    future<std::string> get_locality_name(naming::id_type const& id)
    {
        return async<hpx_get_locality_name_action>(id);
    }
}
