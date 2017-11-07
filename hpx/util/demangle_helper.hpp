//  Copyright (c) 2017 John Biddiscombe
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_DEMANGLE_HELPER_OCT_28_2011_0410PM)
#define HPX_UTIL_DEMANGLE_HELPER_OCT_28_2011_0410PM

#include <hpx/config.hpp>
#include <string>
#include <type_traits>

#if defined(__GNUC__)
#include <cxxabi.h>
#include <stdlib.h>

namespace hpx { namespace debug
{
    // --------------------------------------------------------------------
    // demangle an arbitrary c++ type using gnu utility
    // --------------------------------------------------------------------
    template <typename T>
    class demangle_helper
    {
    public:
        demangle_helper()
          : demangled_(abi::__cxa_demangle(typeid(T).name(), 0, 0, 0))
        {}

        ~demangle_helper()
        {
            free(demangled_);
        }

        char const* type_id() const
        {
            return demangled_ ? demangled_ : typeid(T).name();
        }

    private:
        char* demangled_;
    };

}}

#else

#include <typeinfo>

namespace hpx { namespace util
{
    template <typename T>
    struct demangle_helper
    {
        char const* type_id() const
        {
            return typeid(T).name();
        }
    };
}}

#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace debug
{
    template <typename T>
    struct type_id
    {
        static demangle_helper<T> typeid_;
    };

    template <typename T>
    demangle_helper<T> type_id<T>::typeid_ = demangle_helper<T>();

    // --------------------------------------------------------------------
    // print type information
    // usage : std::cout << print_type<args...>("separator")
    // separator is appended if the number of types > 1
    // --------------------------------------------------------------------
    template <typename T=void>
    inline std::string print_type(const char *delim="")
    {
        return std::string(debug::type_id<T>::typeid_.type_id());;
    }

    template <>
    inline std::string print_type<>(const char *)
    {
        return "void";
    }

    template <typename T, typename ...Args>
    inline typename std::enable_if<sizeof...(Args)!=0, std::string>::type
    print_type(const char *delim="")
    {
        std::string temp(debug::type_id<T>::typeid_.type_id());
        return temp + delim + print_type<Args...>(delim);
    }
}}

#endif

