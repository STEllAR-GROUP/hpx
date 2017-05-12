//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/detail/parcel_await.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace hpx { namespace parcelset { namespace detail {

    parcel_await::parcel_await(parcel&& p, write_handler_type&& f, int archive_flags,
        parcel_await::put_parcel_type pp)
    : base_type(std::move(p), std::move(f), archive_flags, std::move(pp))
    {
    }

    void parcel_await::apply()
    {
        if (apply_single(parcel_))
        {
            done();
        }
    }

    parcels_await::parcels_await(std::vector<parcel>&& p, std::vector<write_handler_type>&& f,
        int archive_flags, parcels_await::put_parcel_type pp)
      : base_type(std::move(p), std::move(f), archive_flags, std::move(pp)),
        idx_(0)
    {
    }

    void parcels_await::apply()
    {
        for (/*idx_*/; idx_ != parcel_.size(); ++idx_)
        {
            if(!apply_single(parcel_[idx_]))
                return;
        }
        done();
    }

}}}
