//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define HPX_NO_VERSION_CHECK

#include <hpx/assertion.hpp>
#include <hpx/testing.hpp>

#include <cstddef>
#include <functional>

namespace hpx { namespace util {
    static test_failure_handler_type test_failure_handler;

    void set_test_failure_handler(test_failure_handler_type f)
    {
        test_failure_handler = f;
    }

    namespace detail
    {
        void fixture::increment(counter_type c)
        {
            if (test_failure_handler)
            {
                test_failure_handler();
            }

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
    }
}}

