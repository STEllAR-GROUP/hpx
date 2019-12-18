//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_DEBUGGING_PRINT_HPP)
#define HPX_DEBUGGING_PRINT_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/thread_data.hpp>

#include <array>
#include <bitset>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux) || defined(linux) || defined(__linux__)
#include <linux/unistd.h>
#include <sys/mman.h>
#define DEBUGGING_PRINT_LINUX
#endif

// ------------------------------------------------------------
// This file provides a simple to use printf style debugging
// tool that can be used on a per file basis to enable ouput.
// It is not intended to be exposed to users, but rather as
// an aid for hpx development.
// ------------------------------------------------------------
// Usage: Instantiate a debug print object at the top of a file
// using a template param of true/false to enable/disable output
// when the template parameter is false, the optimizer will
// not produce code and so the impact is nil.
//
// static hpx::debug::enable_print<true> spq_deb("SUBJECT");
//
// Later in code you may print information using
//
//             spq_deb.debug(str<16>("cleanup_terminated"), "v1"
//                  , "D" , dec<2>(domain_num)
//                  , "Q" , dec<3>(q_index)
//                  , "thread_num", dec<3>(local_num));
//
// various print formatters (dec/hex/str) are supplied to make
// the output regular and aligned for easy parsing/scanning.
//
// In tight loops, huge amounts of debug information might be
// produced, so a simple timer based output is provided
// To instantiate a timed output
//      static auto getnext = spq_deb.make_timer(1
//              , str<16>("get_next_thread"));
// then inside a tight loop
//      spq_deb.timed(getnext, dec<>(thread_num));
// The output will only be produced every N seconds
// ------------------------------------------------------------

// ------------------------------------------------------------
/// \cond NODETAIL
namespace hpx { namespace debug {

    // ------------------------------------------------------------------
    // format as zero padded int
    // ------------------------------------------------------------------
    template <int N = 2, typename T = int>
    struct dec
    {
        dec(const T& v)
          : data(v)
        {
        }
        const T& data;
        friend std::ostream& operator<<(std::ostream& os, const dec<N, T>& d)
        {
            os << std::right << std::setfill('0') << std::setw(N)
               << std::noshowbase << std::dec << d.data;
            return os;
        }
    };

    // ------------------------------------------------------------------
    // format as pointer
    // ------------------------------------------------------------------
    struct ptr
    {
        ptr(void const* v)
          : data(v)
        {
        }
        void const* data;
        friend std::ostream& operator<<(std::ostream& os, const ptr& d)
        {
            os << d.data;
            return os;
        }
    };

    // ------------------------------------------------------------------
    // format as zero padded hex
    // ------------------------------------------------------------------
    template <int N = 4, typename T = int, typename Enable = void>
    struct hex;

    template <int N, typename T>
    struct hex<N, T, typename std::enable_if<!std::is_pointer<T>::value>::type>
    {
        hex(const T& v)
          : data(v)
        {
        }
        const T& data;
        friend std::ostream& operator<<(std::ostream& os, const hex<N, T>& d)
        {
            os << std::right << "0x" << std::setfill('0') << std::setw(N)
               << std::noshowbase << std::hex << d.data;
            return os;
        }
    };

    template <int N, typename T>
    struct hex<N, T, typename std::enable_if<std::is_pointer<T>::value>::type>
    {
        hex(const T& v)
          : data(v)
        {
        }
        const T& data;
        friend std::ostream& operator<<(std::ostream& os, const hex<N, T>& d)
        {
            os << std::right << std::setw(N) << std::noshowbase << std::hex
               << d.data;
            return os;
        }
    };

    // ------------------------------------------------------------------
    // format as binary bits
    // ------------------------------------------------------------------
    template <int N = 8, typename T = int>
    struct bin
    {
        bin(const T& v)
          : data(v)
        {
        }
        const T& data;
        friend std::ostream& operator<<(std::ostream& os, const bin<N, T>& d)
        {
            os << std::bitset<N>(d.data);
            return os;
        }
    };

    // ------------------------------------------------------------------
    // format as padded string
    // ------------------------------------------------------------------
    template <int N = 20>
    struct str
    {
        str(const char* v)
          : data(v)
        {
        }
        const char* data;
        friend std::ostream& operator<<(std::ostream& os, const str<N>& d)
        {
            os << std::left << std::setfill(' ') << std::setw(N) << d.data;
            return os;
        }
    };

    // ------------------------------------------------------------------
    // safely dump thread pointer/description
    // ------------------------------------------------------------------
    template <typename T>
    struct threadinfo
    {
    };

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

#ifdef HPX_HAVE_CXX17_FOLD_EXPRESSIONS
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
               << " cpu " << dec<3, int>(sched_getcpu()) << " ";
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
            static high_resolution_clock::time_point log_t_start =
                high_resolution_clock::now();
            //
            auto now = high_resolution_clock::now();
            auto nowt = duration_cast<microseconds>(now - log_t_start).count();
            //
            os << dec<10>(nowt) << " ";
            return os;
        }

        template <typename TupleType, std::size_t... I>
        void tuple_print(
            std::ostream& os, const TupleType& tup, std::index_sequence<I...>)
        {
            (..., (os << (I == 0 ? "" : " ") << std::get<I>(tup)));
        }

        template <typename... Args>
        void tuple_print(std::ostream& os, const std::tuple<Args...>& tup)
        {
            tuple_print(os, tup, std::make_index_sequence<sizeof...(Args)>());
        }

        template <typename... Args>
        void display(const char* prefix, Args... args)
        {
            // using a temp stream object with a single copy to cout at the end
            // prevents multiple threads from injecting overlapping text
            std::stringstream tempstream;
            tempstream << prefix << detail::current_time_print_helper()
                       << detail::current_thread_print_helper();
            ((tempstream << args << " "), ...);
            tempstream << std::endl;
            std::cout << tempstream.str();
        }

        template <typename... Args>
        void debug(Args... args)
        {
            display("<DEB> ", std::forward<Args>(args)...);
        }

        template <typename... Args>
        void warning(Args... args)
        {
            display("<WAR> ", std::forward<Args>(args)...);
        }

        template <typename... Args>
        void error(Args... args)
        {
            display("<ERR> ", std::forward<Args>(args)...);
        }

        template <typename... Args>
        void timed(Args... args)
        {
            display("<TIM> ", std::forward<Args>(args)...);
        }
    }    // namespace detail
#endif

    template <typename T>
    struct init
    {
        T data;
        init(const T& t)
          : data(t)
        {
        }
        friend std::ostream& operator<<(std::ostream& os, const init<T>& d)
        {
            os << d.data << " ";
            return os;
        }
    };

    template <typename T>
    void set(init<T>& var, const T val)
    {
        var.data = val;
    }

    template <typename... Args>
    struct timed_init
    {
        std::chrono::steady_clock::time_point time_start_;
        double delay_;
        std::tuple<Args...> message_;
        //
        timed_init(double delay, const Args&... args)
          : time_start_(std::chrono::steady_clock::now())
          , delay_(delay)
          , message_(args...)
        {
        }

        bool elapsed(const std::chrono::steady_clock::time_point& now)
        {
            double elapsed_ =
                std::chrono::duration_cast<std::chrono::duration<double>>(
                    now - time_start_)
                    .count();

            if (elapsed_ > delay_)
            {
                time_start_ = now;
                return true;
            }
            return false;
        }

        friend std::ostream& operator<<(
            std::ostream& os, const timed_init<Args...>& ti)
        {
#ifdef HPX_HAVE_CXX17_FOLD_EXPRESSIONS
            detail::tuple_print(os, ti.message_);
#endif
            return os;
        }
    };

    // if fold expressions are not available, all output is disabled
#ifndef HPX_HAVE_CXX17_FOLD_EXPRESSIONS
    template <bool>
    struct enable_print;

    struct disable_print
    {
        HPX_CONSTEXPR disable_print(const char* p) {}
#else
    template <bool enable>
    struct enable_print;

    // when false, debug statements should produce no code
    template <>
    struct enable_print<false>
    {
        HPX_CONSTEXPR enable_print(const char* p) {}
#endif

        HPX_CONSTEXPR bool is_enabled() const
        {
            return false;
        }

        template <typename... Args>
        HPX_CXX14_CONSTEXPR void debug(Args... args) const
        {
        }

        template <typename... Args>
        HPX_CXX14_CONSTEXPR void warning(Args... args) const
        {
        }

        template <typename... Args>
        HPX_CXX14_CONSTEXPR void error(Args... args) const
        {
        }

        template <typename... Args>
        HPX_CXX14_CONSTEXPR void timed(Args... args) const
        {
        }

        template <typename T>
        HPX_CXX14_CONSTEXPR void array(
            const std::string& name, const std::vector<T>& v) const
        {
        }

        template <typename T, std::size_t N>
        HPX_CXX14_CONSTEXPR void array(
            const std::string& name, const std::array<T, N>& v) const
        {
        }

        template <typename Iter>
        HPX_CXX14_CONSTEXPR void array(
            const std::string& name, Iter begin, Iter end) const
        {
        }

        // @todo, return void so that timers have zero footprint when disabled
        template <typename... Args>
        HPX_CONSTEXPR int make_timer(double delay, const Args... args) const
        {
            return 0;
        }
    };

#ifndef HPX_HAVE_CXX17_FOLD_EXPRESSIONS

    template <>
    struct enable_print<false> : public disable_print
    {
        HPX_CONSTEXPR enable_print(const char* p)
          : disable_print(p)
        {
        }
    };

    template <>
    struct enable_print<true> : public disable_print
    {
        HPX_CONSTEXPR enable_print(const char* p)
          : disable_print(p)
        {
        }
    };
#else

    // when true, debug statements produce valid output
    template <>
    struct enable_print<true>
    {
    private:
        const char* prefix_;

    public:
        HPX_CONSTEXPR enable_print(const char* p)
          : prefix_(p)
        {
        }

        HPX_CONSTEXPR bool is_enabled() const
        {
            return true;
        }

        template <typename... Args>
        HPX_CXX14_CONSTEXPR void debug(Args... args)
        {
            detail::debug(prefix_, std::forward<Args>(args)...);
        }
        template <typename... Args>
        HPX_CXX14_CONSTEXPR void warning(Args... args)
        {
            detail::warning(prefix_, std::forward<Args>(args)...);
        }
        template <typename... Args>
        HPX_CXX14_CONSTEXPR void error(Args... args)
        {
            detail::error(prefix_, std::forward<Args>(args)...);
        }
        template <typename... T, typename... Args>
        void timed(timed_init<T...>& init, Args... args)
        {
            auto now = std::chrono::steady_clock::now();
            if (init.elapsed(now))
            {
                detail::timed(prefix_, init, std::forward<Args>(args)...);
            }
        }

        template <typename T>
        void array(const std::string& name, const std::vector<T>& v)
        {
            std::cout << str<20>(name.c_str()) << ": {" << dec<4>(v.size())
                      << "} : ";
            std::copy(std::begin(v), std::end(v),
                std::ostream_iterator<T>(std::cout, ", "));
            std::cout << "\n";
        }

        template <typename T, std::size_t N>
        void array(const std::string& name, const std::array<T, N>& v)
        {
            std::cout << str<20>(name.c_str()) << ": {" << dec<4>(v.size())
                      << "} : ";
            std::copy(std::begin(v), std::end(v),
                std::ostream_iterator<T>(std::cout, ", "));
            std::cout << "\n";
        }

        template <typename Iter>
        void array(const std::string& name, Iter begin, Iter end)
        {
            std::cout << str<20>(name.c_str()) << ": {"
                      << dec<4>(std::distance(begin, end)) << "} : ";
            std::copy(begin, end,
                std::ostream_iterator<
                    typename std::iterator_traits<Iter>::value_type>(
                    std::cout, ", "));
            std::cout << std::endl;
        }

        template <typename T>
        void set(init<T>& var, const T val)
        {
            var.data = val;
        }

        template <typename... Args>
        timed_init<Args...> make_timer(double delay, const Args... args)
        {
            return timed_init<Args...>(delay, args...);
        }
    };
#endif
}}    // namespace hpx::debug
/// \endcond

#endif
