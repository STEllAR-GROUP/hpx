//  Copyright (c) 2017 John Biddiscombe
//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstdlib>
#include <string>
#include <type_traits>
#include <typeinfo>

// --------------------------------------------------------------------
// Always present regardless of compiler : used by serialization code
// --------------------------------------------------------------------
namespace hpx::util::debug {

    template <typename T>
    struct demangle_helper
    {
        [[nodiscard]] static char const* type_id() noexcept
        {
            return typeid(T).name();
        }
    };
}    // namespace hpx::util::debug

#if defined(__GNUG__)

#include <cstdlib>
#include <cxxabi.h>

#include <memory>

// --------------------------------------------------------------------
// if available : demangle an arbitrary c++ type using gnu utility
// --------------------------------------------------------------------
namespace hpx::util::debug {

    template <typename T>
    class cxxabi_demangle_helper
    {
    public:
        cxxabi_demangle_helper()
          : demangled_{abi::__cxa_demangle(
                           typeid(T).name(), nullptr, nullptr, nullptr),
                std::free}
        {
        }

        [[nodiscard]] char const* type_id() const noexcept
        {
            return demangled_ ? demangled_.get() : typeid(T).name();
        }

    private:
        std::unique_ptr<char, void (*)(void*)> demangled_;
    };
}    // namespace hpx::util::debug

#else

namespace hpx::util::debug {

    template <typename T>
    using cxxabi_demangle_helper = demangle_helper<T>;
}    // namespace hpx::util::debug

#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util::debug {

    template <typename T>
    struct type_id
    {
        static demangle_helper<T> typeid_;
    };

    template <typename T>
    demangle_helper<T> type_id<T>::typeid_ = demangle_helper<T>();

#if defined(__GNUG__)
    template <typename T>
    struct cxx_type_id
    {
        static cxxabi_demangle_helper<T> typeid_;
    };

    template <typename T>
    cxxabi_demangle_helper<T> cxx_type_id<T>::typeid_ =
        cxxabi_demangle_helper<T>();
#else
    template <typename T>
    using cxx_type_id = type_id<T>;
#endif

    // --------------------------------------------------------------------
    // print type information
    // usage : std::cout << print_type<args...>("separator")
    // separator is appended if the number of types > 1
    // --------------------------------------------------------------------
    template <typename T = void>
    std::string print_type(char const* = "")
    {
        return std::string(cxx_type_id<T>::typeid_.type_id());
    }

    template <>
    inline std::string print_type<>(char const*)
    {
        return "void";
    }

    template <typename T, typename... Args>
    std::enable_if_t<sizeof...(Args) != 0, std::string> print_type(
        char const* delim = "")
    {
        std::string const temp(cxx_type_id<T>::typeid_.type_id());
        return temp + delim + print_type<Args...>(delim);
    }
}    // namespace hpx::util::debug
