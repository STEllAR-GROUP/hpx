//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/functional/unique_function.hpp>

#include <cstdint>
#include <vector>

namespace hpx { namespace parcelset { namespace detail
{
    using put_parcel_type = hpx::util::unique_function_nonser<
            void(parcel&&, write_handler_type&&)
        >;

    void HPX_EXPORT parcel_await_apply(parcel&& p, write_handler_type&& f,
        std::uint32_t archive_flags, put_parcel_type pp);

    using put_parcels_type = hpx::util::unique_function_nonser<
            void(std::vector<parcel>&&, std::vector<write_handler_type>&&)
        >;

    void HPX_EXPORT parcels_await_apply(std::vector<parcel>&& p,
        std::vector<write_handler_type>&& f, std::uint32_t archive_flags,
        put_parcels_type pp);
}}}

#endif
