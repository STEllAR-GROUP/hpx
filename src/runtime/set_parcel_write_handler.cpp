//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/runtime_distributed.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/runtime/set_parcel_write_handler.hpp>
#include <hpx/runtime/runtime_fwd.hpp>

namespace hpx
{
    parcel_write_handler_type set_parcel_write_handler(
        parcel_write_handler_type const& f)
    {
        runtime_distributed* rt = get_runtime_distributed_ptr();
        if (nullptr != rt)
            return rt->get_parcel_handler().set_write_handler(f);

        HPX_THROW_EXCEPTION(invalid_status,
            "hpx::set_default_parcel_write_handler",
            "the runtime system is not operational at this point");
    }
}

#endif
