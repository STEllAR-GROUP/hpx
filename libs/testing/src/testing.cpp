//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define HPX_NO_VERSION_CHECK

#include <hpx/assertion.hpp>
#include <hpx/testing.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <functional>

#include <boost/io/ios_state.hpp>

namespace hpx { namespace util
{
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
                ++sanity_failures_;
                return;
            case counter_test:
                ++test_failures_;
                return;
            default:
                break;
            }
            HPX_ASSERT(false);
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
                break;
            }
            HPX_ASSERT(false);
            return std::size_t(-1);
        }

        fixture global_fixture{std::cerr};
    }

    ////////////////////////////////////////////////////////////////////////////
    int report_errors(std::ostream& stream)
    {
        std::size_t sanity = detail::global_fixture.get(counter_sanity),
                    test = detail::global_fixture.get(counter_test);
        if (sanity == 0 && test == 0)
            return 0;

        else
        {
            boost::io::ios_flags_saver ifs(stream);
            stream << sanity << " sanity check"    //-V128
                   << ((sanity == 1) ? " and " : "s and ") << test << " test"
                   << ((test == 1) ? " failed." : "s failed.") << std::endl;
            return 1;
        }
    }

    void print_cdash_timing(const char* name, double time)
    {
        // use stringstream followed by single cout for better multi-threaded
        // output
        std::stringstream temp;
        temp << "<DartMeasurement name=\"" << name << "\" "
             << "type=\"numeric/double\">" << time << "</DartMeasurement>";
        std::cout << temp.str() << std::endl;
    }

    void print_cdash_timing(const char* name, std::uint64_t time)
    {
        print_cdash_timing(name, time / 1e9);
    }
}}
