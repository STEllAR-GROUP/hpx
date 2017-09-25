//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define HPX_NO_VERSION_CHECK

#include <hpx/runtime/config_entry.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/debugging.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <iostream>

namespace hpx { namespace util { namespace detail
{
    /// Attach the debugger if this is enabled in the config
    static void may_attach_debugger()
    {
        if (get_config_entry("hpx.attach-debugger", "") == "test-failure")
        {
            attach_debugger();
        }
    }

    void fixture::increment(counter_type c)
    {
        may_attach_debugger();

        switch (c)
        {
            case counter_sanity:
                ++sanity_failures_; return;
            case counter_test:
                ++test_failures_; return;
            default:
                { HPX_ASSERT(false); return; }
        }
    }

    std::size_t fixture::get(counter_type c) const
    {
        switch (c)
        {
            case counter_sanity:
                return sanity_failures_;
            case counter_test:
                return test_failures_;
            default:
                { HPX_ASSERT(false); return 0; }
        }
    }

    fixture global_fixture{std::cerr};
}}}

