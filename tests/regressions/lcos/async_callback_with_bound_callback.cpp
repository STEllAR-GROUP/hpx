//  Copyright 2015 (c) Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/include/async.hpp>
#include <hpx/testing.hpp>

#if defined(HPX_HAVE_NETWORKING)
void async_callback(uint64_t index, boost::system::error_code const& ec,
    hpx::parcelset::parcel const& p)
{
}
#else
void async_callback(uint64_t index)
{
}
#endif

void func()
{
}
HPX_PLAIN_ACTION(func);

int main()
{
    for (hpx::id_type const& id: hpx::find_all_localities())
    {
        uint64_t buffer_index = 0;
#if defined(HPX_HAVE_NETWORKING)
        hpx::future<void> f = hpx::async_cb(func_action(), id,
            hpx::util::bind(&async_callback, buffer_index,
                hpx::util::placeholders::_1, hpx::util::placeholders::_2));
#else
        hpx::future<void> f = hpx::async_cb(func_action(), id,
            hpx::util::bind(&async_callback, buffer_index));
#endif
        f.get();
    }
    return hpx::util::report_errors();
}
