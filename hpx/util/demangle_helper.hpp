//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_DEMANGLE_HELPER_OCT_28_2011_0410PM)
#define HPX_UTIL_DEMANGLE_HELPER_OCT_28_2011_0410PM

#include <hpx/config.hpp>

// disable the code specific to gcc for now as this causes problems
// (see #811: simple_central_tuplespace_client run error)
#if 0 // defined(__GNUC__)
#include <cxxabi.h>

namespace hpx { namespace util
{
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
namespace hpx { namespace util
{
    template <typename T>
    struct type_id
    {
        static demangle_helper<T> typeid_;
    };

    template <typename T>
    demangle_helper<T> type_id<T>::typeid_ = demangle_helper<T>();
}}

#endif

