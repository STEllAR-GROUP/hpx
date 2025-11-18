////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/preprocessor.hpp>
#include <hpx/modules/thread_support.hpp>
#include <hpx/modules/util.hpp>
#include <hpx/testing/performance.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <ostream>

namespace hpx::util {

    using test_failure_handler_type = hpx::function<void()>;

    HPX_CORE_EXPORT void set_test_failure_handler(test_failure_handler_type f);

    enum class counter_type
    {
        sanity,
        test
    };

    namespace detail {

        struct fixture
        {
        public:
            using mutex_type = hpx::util::detail::spinlock;

        private:
            std::ostream& stream_;
            mutex_type mutex_;

        public:
            explicit fixture(std::ostream& stream) noexcept
              : stream_(stream)
            {
            }

            HPX_CORE_EXPORT static void increment(counter_type c) noexcept;

            HPX_CORE_EXPORT static std::size_t get(counter_type c) noexcept;

            template <typename T>
            bool check_(char const* file, int line, char const* function,
                counter_type c, T const& t, char const* msg)
            {
                if (!t)
                {
                    std::lock_guard<mutex_type> l(mutex_);
                    hpx::util::ios_flags_saver ifs(stream_);
                    stream_ << file << "(" << line << "): " << msg
                            << " failed in function '" << function << "'"
                            << std::endl;
                    increment(c);
                    return false;
                }
                return true;
            }

            template <typename T, typename U>
            bool check_equal(char const* file, int line, char const* function,
                counter_type c, T const& t, U const& u, char const* msg)
            {
                if (!(t == u))
                {
                    std::lock_guard<mutex_type> l(mutex_);
                    hpx::util::ios_flags_saver ifs(stream_);
                    stream_ << file << "(" << line << "): " << msg
                            << " failed in function '" << function << "': "
                            << "'" << t << "' != '" << u << "'" << std::endl;
                    increment(c);
                    return false;
                }
                return true;
            }

            template <typename T, typename U>
            bool check_not_equal(char const* file, int line,
                char const* function, counter_type c, T const& t, U const& u,
                char const* msg)
            {
                if (!(t != u))
                {
                    std::lock_guard<mutex_type> l(mutex_);
                    hpx::util::ios_flags_saver ifs(stream_);
                    stream_ << file << "(" << line << "): " << msg
                            << " failed in function '" << function << "': "
                            << "'" << t << "' != '" << u << "'" << std::endl;
                    increment(c);
                    return false;
                }
                return true;
            }

            template <typename T, typename U>
            bool check_less(char const* file, int line, char const* function,
                counter_type c, T const& t, U const& u, char const* msg)
            {
                if (!(t < u))
                {
                    std::lock_guard<mutex_type> l(mutex_);
                    hpx::util::ios_flags_saver ifs(stream_);
                    stream_ << file << "(" << line << "): " << msg
                            << " failed in function '" << function << "': "
                            << "'" << t << "' >= '" << u << "'" << std::endl;
                    increment(c);
                    return false;
                }
                return true;
            }

            template <typename T, typename U>
            bool check_less_equal(char const* file, int line,
                char const* function, counter_type c, T const& t, U const& u,
                char const* msg)
            {
                if (!(t <= u))
                {
                    std::lock_guard<mutex_type> l(mutex_);
                    hpx::util::ios_flags_saver ifs(stream_);
                    stream_ << file << "(" << line << "): " << msg
                            << " failed in function '" << function << "': "
                            << "'" << t << "' > '" << u << "'" << std::endl;
                    increment(c);
                    return false;
                }
                return true;
            }

            template <typename T, typename U, typename V>
            bool check_range(char const* file, int line, char const* function,
                counter_type c, T const& t, U const& u, V const& v,
                char const* msg)
            {
                if (!(t >= u && t <= v))
                {
                    std::lock_guard<mutex_type> l(mutex_);
                    hpx::util::ios_flags_saver ifs(stream_);
                    if (!(t >= u))
                    {
                        stream_ << file << "(" << line << "): " << msg
                                << " failed in function '" << function << "': "
                                << "'" << t << "' < '" << u << "'" << std::endl;
                    }
                    else
                    {
                        stream_ << file << "(" << line << "): " << msg
                                << " failed in function '" << function << "': "
                                << "'" << t << "' > '" << v << "'" << std::endl;
                    }
                    increment(c);
                    return false;
                }
                return true;
            }
        };

        HPX_CORE_EXPORT fixture& global_fixture() noexcept;

    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////////
    HPX_CORE_EXPORT int report_errors();
    HPX_CORE_EXPORT int report_errors(std::ostream& stream);
    HPX_CORE_EXPORT void print_cdash_timing(char const* name, double time);
    HPX_CORE_EXPORT void print_cdash_timing(
        char const* name, std::uint64_t time);
}    // namespace hpx::util
