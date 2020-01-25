//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

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
              : data_(v)
            {
            }

            T const& data_;

            friend std::ostream& operator<<(
                std::ostream& os, dec<N, T> const& d)
            {
                os << std::right << std::setfill('0') << std::setw(N)
                   << std::noshowbase << std::dec << d.data_;
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
          : data_(v)
        {
        }
        ptr(std::uintptr_t const v)
          : data_(reinterpret_cast<void const*>(v))
        {
        }
        void const* data_;
        friend std::ostream& operator<<(std::ostream& os, const ptr& d)
        {
            os << d.data_;
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
              : data_(v)
            {
            }
            const T& data_;
            friend std::ostream& operator<<(
                std::ostream& os, const hex<N, T>& d)
            {
                os << std::right << "0x" << std::setfill('0') << std::setw(N)
                   << std::noshowbase << std::hex << d.data_;
                return os;
            }
        };

        template <int N, typename T>
        struct hex<N, T,
            typename std::enable_if<std::is_pointer<T>::value>::type>
        {
            hex(const T& v)
              : data_(v)
            {
            }
            const T& data_;
            friend std::ostream& operator<<(
                std::ostream& os, const hex<N, T>& d)
            {
                os << std::right << std::setw(N) << std::noshowbase << std::hex
                   << d.data_;
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
              : data_(v)
            {
            }
            const T& data_;
            friend std::ostream& operator<<(
                std::ostream& os, const bin<N, T>& d)
            {
                os << std::bitset<N>(d.data_);
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
          : data_(v)
        {
        }
        const char* data_;
        friend std::ostream& operator<<(std::ostream& os, const str<N>& d)
        {
            os << std::left << std::setfill(' ') << std::setw(N) << d.data_;
            return os;
        }
    };

    // ------------------------------------------------------------------
    // format as ip address
    // ------------------------------------------------------------------
    struct ipaddr
    {
        ipaddr(const void* a)
          : data_(reinterpret_cast<const uint8_t*>(a))
        {
        }
        const uint8_t* data_;
        friend std::ostream& operator<<(std::ostream& os, const ipaddr& p)
        {
            os << std::dec << int((reinterpret_cast<const uint8_t*>(&p))[0])
               << "." << int((reinterpret_cast<const uint8_t*>(&p))[1]) << "."
               << int((reinterpret_cast<const uint8_t*>(&p))[2]) << "."
               << int((reinterpret_cast<const uint8_t*>(&p))[3]);
            return os;
        }
    };

    // ------------------------------------------------------------------
    // helper fuction for printing CRC32
    // ------------------------------------------------------------------
    inline uint32_t crc32(const void* address, size_t length)
    {
        //        boost::crc_32_type result;
        //        result.process_bytes(address, length);
        //        return result.checksum();
        return 0;
    }

    // ------------------------------------------------------------------
    // helper fuction for printing short memory dump and crc32
    // useful for debugging corruptions in buffers during parcelport
    // rma or other transfers
    // ------------------------------------------------------------------
    struct mem_crc32
    {
        mem_crc32(const void* a, std::size_t len, const char* txt)
          : addr_(reinterpret_cast<const uint64_t*>(a))
          , len_(len)
          , txt_(txt)
        {
        }
        const uint64_t* addr_;
        const std::size_t len_;
        const char* txt_;
        friend std::ostream& operator<<(std::ostream& os, const mem_crc32& p)
        {
            const uint64_t* uintBuf = static_cast<const uint64_t*>(p.addr_);
            os << "Memory:";
            os << " address " << hpx::debug::ptr(p.addr_) << " length "
               << hpx::debug::hex<8>(p.len_) << " CRC32:\n"
               << hpx::debug::hex<8>(crc32(p.addr_, p.len_)) << " ";
            for (size_t i = 0; i < (std::min)(p.len_ / 8, size_t(128)); i++)
            {
                os << hpx::debug::hex<16>(*uintBuf++) << " ";
            }
            os << " : " << p.txt_;
            return os;
        }
    };

    namespace detail {

#ifdef HPX_HAVE_CXX17_FOLD_EXPRESSIONS
        template <typename TupleType, std::size_t... I>
        void tuple_print(
            std::ostream& os, const TupleType& tup, std::index_sequence<I...>)
        {
            (..., (os << (I == 0 ? "" : " ") << std::get<I>(tup)));
        }

        template <typename... Args>
        void tuple_print(std::ostream& os,
            const std::tuple<std::forward<Args>(args)...>& tup)
        {
            tuple_print(os, tup, std::make_index_sequence<sizeof...(Args)>());
        }

#else
        // C++14 version
        // helper function to print a tuple of any size
        template <typename TupleType, std::size_t N>
        struct tuple_printer
        {
            static void print(std::ostream& os, const TupleType& t)
            {
                tuple_printer<TupleType, N - 1>::print(os, t);
                os << ", " << std::get<N - 1>(t);
            }
        };

        template <typename TupleType>
        struct tuple_printer<TupleType, 1>
        {
            static void print(std::ostream& os, const TupleType& t)
            {
                os << std::get<0>(t);
            }
        };

        template <class... Args>
        void tuple_print(std::ostream& os, const std::tuple<Args...>& t)
        {
            tuple_printer<decltype(t), sizeof...(Args)>::print(os, t);
        }

        template <typename Arg, typename... Args>
        void variadic_print(std::ostream& os, const Arg &arg, const Args&... args)
        {
            os << arg;
            using expander = int[];
            (void) expander{0, (void(os << ' ' << args), 0)...};
        }

        template <class... Args>
        void tuple_print(std::ostream& os, const Args&... args)
        {
            variadic_print(os, args...);
        }

#endif

    }

    namespace detail {

        template <typename... Args>
        void display(const char* prefix, const Args&... args);

        template <typename... Args>
        void debug(const Args&... args)
        {
            display("<DEB> ", args...);
        }

        template <typename... Args>
        void warning(const Args&... args)
        {
            display("<WAR> ", args...);
        }

        template <typename... Args>
        void error(const Args&... args)
        {
            display("<ERR> ", args...);
        }

        template <typename... Args>
        void trace(const Args&... args)
        {
            display("<TRC> ", args...);
        }

        template <typename... Args>
        void timed(const Args... args)
        {
            display("<TIM> ", args...);
        }
    }    // namespace detail

    template <typename T>
    struct init
    {
        T data_;
        init(const T& t)
          : data_(t)
        {
        }
        friend std::ostream& operator<<(std::ostream& os, const init<T>& d)
        {
            os << d.data_ << " ";
            return os;
        }
    };

    template <typename T>
    void set(init<T>& var, const T val)
    {
        var.data_ = val;
    }

    template <typename... Args>
    struct timed_init
    {
        mutable std::chrono::steady_clock::time_point time_start_;
        double delay_;
        std::tuple<std::decay_t<Args>...> message_;
        //
        timed_init(double delay, Args&... args)
          : time_start_(std::chrono::steady_clock::now())
          , delay_(delay)
          , message_(args...)
        {
        }

        bool elapsed(const std::chrono::steady_clock::time_point& now) const
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
            detail::tuple_print(os, ti.message_);
            return os;
        }
    };

    template <bool enable>
    struct enable_print;

    // when false, debug statements should produce no code
    template <>
    struct enable_print<false>
    {
        constexpr enable_print(const char*) {}

        constexpr bool is_enabled() const
        {
            return false;
        }

        template <typename... Args>
        constexpr void debug(const Args&...) const
        {
        }

        template <typename... Args>
        constexpr void warning(const Args&...) const
        {
        }

        template <typename... Args>
        constexpr void trace(const Args&...) const
        {
        }

        template <typename... Args>
        constexpr void error(const Args&...) const
        {
        }

        template <typename... Args>
        constexpr void timed(const Args&...) const
        {
        }

        template <typename T>
        constexpr void array(
            const std::string& name, const std::vector<T>&) const
        {
        }

        template <typename T, std::size_t N>
        constexpr void array(
            const std::string& name, const std::array<T, N>&) const
        {
        }

        template <typename Iter>
        constexpr void array(
            const std::string& name, Iter begin, Iter end) const
        {
        }

        template <typename T, typename... Args>
        constexpr bool declare_variable(const Args&...) const
        {
            return true;
        }

        // @todo, return void so that timers have zero footprint when disabled
        template <typename... Args>
        constexpr int make_timer(const double, const Args&...) const
        {
            return 0;
        }
    };

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
        constexpr void debug(const Args&... args) const
        {
            detail::debug(prefix_, args...);
        }

        template <typename... Args>
        constexpr void warning(const Args&... args) const
        {
            detail::warning(prefix_, args...);
        }

        template <typename... Args>
        constexpr void trace(const Args&... args) const
        {
            detail::trace(prefix_, args...);
        }

        template <typename... Args>
        constexpr void error(const Args&... args) const
        {
            detail::error(prefix_, args...);
        }

        template <typename... T, typename... Args>
        void timed(timed_init<T...>& init, const Args&... args) const
        {
            auto now = std::chrono::steady_clock::now();
            if (init.elapsed(now))
            {
                detail::timed(prefix_, init, args...);
            }
        }

        template <typename T>
        void array(const std::string& name, const std::vector<T>& v) const
        {
            std::cout << str<20>(name.c_str()) << ": {"
                      << debug::dec<4>(v.size()) << "} : ";
            std::copy(std::begin(v), std::end(v),
                std::ostream_iterator<T>(std::cout, ", "));
            std::cout << "\n";
        }

        template <typename T, std::size_t N>
        void array(const std::string& name, const std::array<T, N>& v) const
        {
            std::cout << str<20>(name.c_str()) << ": {"
                      << debug::dec<4>(v.size()) << "} : ";
            std::copy(std::begin(v), std::end(v),
                std::ostream_iterator<T>(std::cout, ", "));
            std::cout << "\n";
        }

        template <typename Iter>
        void array(const std::string& name, Iter begin, Iter end) const
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
        void set(init<T>& var, const T val) const
        {
            var.data_ = val;
        }

        template <typename T, typename... Args>
        T declare_variable(const Args&... args) const
        {
            return T(args...);
        }

        template <typename... Args>
        timed_init<Args...> make_timer(const double delay, Args... args)
        {
            return timed_init<Args...>(delay, args...);
        }
    };

}}    // namespace hpx::debug
/// \endcond
