//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_DEBUGGING_PRINT_HPP)
#define HPX_DEBUGGING_PRINT_HPP

#include <hpx/config.hpp>

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
// tool that can be used on a per file basis to enable output.
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
    namespace detail {

        template <int N, typename T>
        struct dec
        {
            dec(T const& v)
              : data(v)
            {
            }

            T const& data;

            friend std::ostream& operator<<(
                std::ostream& os, dec<N, T> const& d)
            {
                os << std::right << std::setfill('0') << std::setw(N)
                   << std::noshowbase << std::dec << d.data;
                return os;
            }
        };
    }    // namespace detail

    template <int N = 2, typename T>
    detail::dec<N, T> dec(T const& v)
    {
        return detail::dec<N, T>(v);
    }

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
    namespace detail {

        template <int N = 4, typename T = int, typename Enable = void>
        struct hex;

        template <int N, typename T>
        struct hex<N, T,
            typename std::enable_if<!std::is_pointer<T>::value>::type>
        {
            hex(const T& v)
              : data(v)
            {
            }
            const T& data;
            friend std::ostream& operator<<(
                std::ostream& os, const hex<N, T>& d)
            {
                os << std::right << "0x" << std::setfill('0') << std::setw(N)
                   << std::noshowbase << std::hex << d.data;
                return os;
            }
        };

        template <int N, typename T>
        struct hex<N, T,
            typename std::enable_if<std::is_pointer<T>::value>::type>
        {
            hex(const T& v)
              : data(v)
            {
            }
            const T& data;
            friend std::ostream& operator<<(
                std::ostream& os, const hex<N, T>& d)
            {
                os << std::right << std::setw(N) << std::noshowbase << std::hex
                   << d.data;
                return os;
            }
        };
    }    // namespace detail

    template <int N = 4, typename T>
    detail::hex<N, T> hex(T const& v)
    {
        return detail::hex<N, T>(v);
    }

    // ------------------------------------------------------------------
    // format as binary bits
    // ------------------------------------------------------------------
    namespace detail {

        template <int N = 8, typename T = int>
        struct bin
        {
            bin(const T& v)
              : data(v)
            {
            }
            const T& data;
            friend std::ostream& operator<<(
                std::ostream& os, const bin<N, T>& d)
            {
                os << std::bitset<N>(d.data);
                return os;
            }
        };
    }    // namespace detail

    template <int N = 8, typename T>
    detail::bin<N, T> bin(T const& v)
    {
        return detail::bin<N, T>(v);
    }

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

#ifdef HPX_HAVE_CXX17_FOLD_EXPRESSIONS
    namespace detail {
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
        void display(const char* prefix, Args... args);

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
        constexpr disable_print(const char* p) {}
#else
    template <bool enable>
    struct enable_print;

    // when false, debug statements should produce no code
    template <>
    struct enable_print<false>
    {
        constexpr enable_print(const char* p) {}
#endif

        constexpr bool is_enabled() const
        {
            return false;
        }

        template <typename... Args>
        constexpr void debug(Args... args) const
        {
        }

        template <typename... Args>
        constexpr void warning(Args... args) const
        {
        }

        template <typename... Args>
        constexpr void error(Args... args) const
        {
        }

        template <typename... Args>
        constexpr void timed(Args... args) const
        {
        }

        template <typename T>
        constexpr void array(
            const std::string& name, const std::vector<T>& v) const
        {
        }

        template <typename T, std::size_t N>
        constexpr void array(
            const std::string& name, const std::array<T, N>& v) const
        {
        }

        template <typename Iter>
        constexpr void array(
            const std::string& name, Iter begin, Iter end) const
        {
        }

        // @todo, return void so that timers have zero footprint when disabled
        template <typename... Args>
        constexpr int make_timer(double delay, const Args... args) const
        {
            return 0;
        }
    };

#ifndef HPX_HAVE_CXX17_FOLD_EXPRESSIONS

    template <>
    struct enable_print<false> : public disable_print
    {
        constexpr enable_print(const char* p)
          : disable_print(p)
        {
        }
    };

    template <>
    struct enable_print<true> : public disable_print
    {
        constexpr enable_print(const char* p)
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
        enable_print(const char* p)
          : prefix_(p)
        {
        }

        constexpr bool is_enabled() const
        {
            return true;
        }

        template <typename... Args>
        constexpr void debug(Args... args)
        {
            detail::debug(prefix_, std::forward<Args>(args)...);
        }
        template <typename... Args>
        constexpr void warning(Args... args)
        {
            detail::warning(prefix_, std::forward<Args>(args)...);
        }
        template <typename... Args>
        constexpr void error(Args... args)
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
            std::cout << str<20>(name.c_str()) << ": {"
                      << debug::dec<4>(v.size()) << "} : ";
            std::copy(std::begin(v), std::end(v),
                std::ostream_iterator<T>(std::cout, ", "));
            std::cout << "\n";
        }

        template <typename T, std::size_t N>
        void array(const std::string& name, const std::array<T, N>& v)
        {
            std::cout << str<20>(name.c_str()) << ": {"
                      << debug::dec<4>(v.size()) << "} : ";
            std::copy(std::begin(v), std::end(v),
                std::ostream_iterator<T>(std::cout, ", "));
            std::cout << "\n";
        }

        template <typename Iter>
        void array(const std::string& name, Iter begin, Iter end)
        {
            std::cout << str<20>(name.c_str()) << ": {"
                      << debug::dec<4>(std::distance(begin, end)) << "} : ";
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
