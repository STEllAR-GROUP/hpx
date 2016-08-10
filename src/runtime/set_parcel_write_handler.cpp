//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/set_parcel_write_handler.hpp>

namespace hpx
{
    HPX_API_EXPORT parcel_write_handler_type set_parcel_write_handler(
        parcel_write_handler_type const& f)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
            return rt->get_parcel_handler().set_write_handler(f);

        HPX_THROW_EXCEPTION(invalid_status,
            "hpx::set_default_parcel_write_handler",
            "the runtime system is not operational at this point");
    }
}
