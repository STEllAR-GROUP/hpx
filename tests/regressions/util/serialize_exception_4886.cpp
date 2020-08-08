//  Copyright (c) 2020 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/modules/testing.hpp>

#include <exception>

void thrower()
{
    throw std::exception();
}
HPX_PLAIN_ACTION(thrower, thrower_action);

int main()
{
    hpx::id_type to_call = hpx::find_here();

    std::vector<hpx::id_type> locales = hpx::find_all_localities();
    for (auto const& id : locales)
    {
        if (id != to_call)
        {
            to_call = id;
            break;
        }
    }

    thrower_action ac;
    hpx::future<void> f = hpx::async(ac, to_call);

    bool caught_exception = false;
    hpx::future<void> g =
        f.then(hpx::launch::sync, [&caught_exception](hpx::future<void>&& f) {
            caught_exception = f.has_exception();
        });

    g.get();
    HPX_TEST(caught_exception);

    return hpx::util::report_errors();
}
