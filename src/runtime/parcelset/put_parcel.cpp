//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/parcelset/put_parcel.hpp>

namespace hpx { namespace parcelset { namespace detail
{
    void put_parcel_handler::operator()(parcel&& p) const
    {
        parcelset::parcelhandler& ph =
            hpx::get_runtime().get_parcel_handler();
        ph.put_parcel(std::move(p));
    }
}}}

