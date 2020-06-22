//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/debugging/print.hpp>
#include <hpx/threading_base/thread_data.hpp>

#include <iostream>
#include <sstream>
#include <thread>

// ------------------------------------------------------------
/// \cond NODETAIL
namespace hpx { namespace debug {
    // ------------------------------------------------------------------
    // safely dump thread pointer/description
    // ------------------------------------------------------------------
    template <typename T>
    struct threadinfo
    {
    };

    // ------------------------------------------------------------------
    // safely dump thread pointer/description
    // ------------------------------------------------------------------
    template <>
    struct threadinfo<threads::thread_data*>
    {
        threadinfo(const threads::thread_data* v)
          : data(v)
        {
        }
        const threads::thread_data* data;
        friend std::ostream& operator<<(std::ostream& os, const threadinfo& d)
        {
            os << ptr(d.data) << " \""
               << ((d.data != nullptr) ? d.data->get_description() : "nullptr")
               << "\"";
            return os;
        }
    };

    template <>
    struct threadinfo<threads::thread_id_type*>
    {
        threadinfo(const threads::thread_id_type* v)
          : data(v)
        {
        }
        const threads::thread_id_type* data;
        friend std::ostream& operator<<(std::ostream& os, const threadinfo& d)
        {
            if (d.data == nullptr)
                os << "nullptr";
            else
                os << threadinfo<threads::thread_data*>(
                    get_thread_id_data(*d.data));
            return os;
        }
    };

    template <>
    struct threadinfo<hpx::threads::thread_init_data>
    {
        threadinfo(const hpx::threads::thread_init_data& v)
          : data(v)
        {
        }
        const hpx::threads::thread_init_data& data;
        friend std::ostream& operator<<(std::ostream& os, const threadinfo& d)
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            os << std::left << " \"" << d.data.description.get_description()
               << "\"";
#else
            os << "??? " << /*hex<8,uintptr_t>*/ (uintptr_t(&d.data));
#endif
            return os;
        }
    };

    namespace detail {
        // ------------------------------------------------------------------
        // helper class for printing thread ID, either std:: or hpx::
        // ------------------------------------------------------------------
        struct current_thread_print_helper
        {
        };

        inline std::ostream& operator<<(
            std::ostream& os, const current_thread_print_helper&)
        {
            if (hpx::threads::get_self_id() == hpx::threads::invalid_thread_id)
            {
                os << "-------------- ";
            }
            else
            {
                hpx::threads::thread_data* dummy =
                    hpx::threads::get_self_id_data();
                os << dummy << " ";
            }
            os << hex<12, std::thread::id>(std::this_thread::get_id())
#ifdef DEBUGGING_PRINT_LINUX
               << " cpu " << debug::dec<3, int>(sched_getcpu()) << " ";
#else
               << " cpu "
               << "--- ";
#endif
            return os;
        }

        // ------------------------------------------------------------------
        // helper class for printing time since start
        // ------------------------------------------------------------------
        struct current_time_print_helper
        {
        };

        inline std::ostream& operator<<(
            std::ostream& os, const current_time_print_helper&)
        {
            using namespace std::chrono;
            static steady_clock::time_point log_t_start = steady_clock::now();
            //
            auto now = steady_clock::now();
            auto nowt = duration_cast<microseconds>(now - log_t_start).count();
            //
            os << debug::dec<10>(nowt) << " ";
            return os;
        }

#ifdef HPX_HAVE_CXX17_FOLD_EXPRESSIONS
        template <typename... Args>
        void display(const char* prefix, const Args&... args)
        {
            // using a temp stream object with a single copy to cout at the end
            // prevents multiple threads from injecting overlapping text
            std::stringstream tempstream;
            tempstream << prefix << detail::current_time_print_helper()
                       << detail::current_thread_print_helper()
                       << detail::hostname_print_helper();
            ((tempstream << args << " "), ...);
            tempstream << std::endl;
            std::cout << tempstream.str();
        }

#else
        template <typename... Args>
        void display(const char* prefix, const Args&... args)
        {
            // using a temp stream object with a single copy to cout at the end
            // prevents multiple threads from injecting overlapping text
            std::stringstream tempstream;
            tempstream << prefix << detail::current_time_print_helper()
                       << detail::current_thread_print_helper()
                       << detail::hostname_print_helper();
            variadic_print(tempstream, args...);
            tempstream << std::endl;
            std::cout << tempstream.str();
        }
#endif

    }    // namespace detail
}}       // namespace hpx::debug
/// \endcond
