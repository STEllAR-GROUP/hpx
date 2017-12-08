//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_PARCEL_AWAIT_HPP
#define HPX_PARCELSET_PARCEL_AWAIT_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/util/unique_function.hpp>

#include <vector>

namespace hpx { namespace parcelset { namespace detail
{
    typedef hpx::util::unique_function_nonser<
            void(parcel&&, write_handler_type&&)
        > put_parcel_type;

    void HPX_EXPORT parcel_await_apply(parcel&& p, write_handler_type&& f,
        int archive_flags, put_parcel_type pp);

    typedef hpx::util::unique_function_nonser<
            void(std::vector<parcel>&&, std::vector<write_handler_type>&&)
        > put_parcels_type;

    void HPX_EXPORT parcels_await_apply(std::vector<parcel>&& p,
        std::vector<write_handler_type>&& f, int archive_flags,
        put_parcels_type pp);
}}}

#endif
