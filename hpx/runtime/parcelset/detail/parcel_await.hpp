//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_PARCEL_AWAIT_HPP
#define HPX_PARCELSET_PARCEL_AWAIT_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/actions_fwd.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/serialization/detail/preprocess.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace hpx { namespace parcelset { namespace detail {
    struct parcel_await
      : std::enable_shared_from_this<parcel_await>
    {
        typedef hpx::util::unique_function_nonser<void(parcel&&)> put_parcel_type;

        parcel_await(parcel&& p, int archive_flags, put_parcel_type pp);

        parcel_await(std::vector<parcel>&& parcels, int archive_flags,
            put_parcel_type pp);

        HPX_EXPORT void apply();

        put_parcel_type put_parcel_;
        std::vector<parcel> parcels_;
        int archive_flags_;
        std::size_t idx_;
    };
}}}

#endif
