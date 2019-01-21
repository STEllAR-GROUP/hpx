//  Copyright (c) 2019 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/assert/source_location.hpp>

#include <iostream>

namespace hpx { namespace assertion {
    namespace {
        assertion_handler& get_handler()
        {
            static assertion_handler handler = nullptr;
            return handler;
        }
    }

    void set_assertion_handler(assertion_handler handler)
    {
        if (get_handler() == nullptr)
        {
            get_handler() = handler;
        }
    }

    namespace detail {
        void handle_assert(source_location const& loc, const char* expr,
            std::string const& msg)
        {
            if (get_handler() == nullptr)
            {
                std::cerr << loc << ": Assertion '" << expr << "' failed";
                if (!msg.empty())
                {
                    std::cerr << " (" << msg << ")\n";
                }
                else
                {
                    std::cerr << '\n';
                }
                std::abort();
            }
            get_handler()(loc, expr, msg);
        }
    }
}}
