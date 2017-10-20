//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_FUNCTION_REGISTRATION_HPP
#define HPX_UTIL_DETAIL_FUNCTION_REGISTRATION_HPP

#include <hpx/config.hpp>
#include <hpx/util/demangle_helper.hpp>
#include <hpx/util/detail/pp/stringize.hpp>
#include <hpx/util/detail/pp/strip_parens.hpp>

#include <type_traits>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable, typename T>
    struct get_function_name_declared
      : std::false_type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable, typename F>
    struct get_function_name_impl
    {
        static char const* call()
#ifdef HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION
        {
            return util::type_id<F>::typeid_.type_id();
        }
#else
        = delete;
#endif
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable, typename F>
    char const* get_function_name()
    {
        return get_function_name_impl<VTable, F>::call();
    }
}}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_DECLARE_GET_FUNCTION_NAME(VTable, F, Name)                        \
    namespace hpx { namespace util { namespace detail {                       \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_function_name<                                        \
            VTable, std::decay<HPX_PP_STRIP_PARENS(F)>::type>();              \
                                                                              \
        template <>                                                           \
        struct get_function_name_declared<                                    \
            VTable, std::decay<HPX_PP_STRIP_PARENS(F)>::type                  \
        > : std::true_type                                                    \
        {};                                                                   \
    }}}                                                                       \
/**/

#define HPX_DEFINE_GET_FUNCTION_NAME(VTable, F, Name)                         \
    namespace hpx { namespace util { namespace detail {                       \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_function_name<                                        \
            VTable, std::decay<HPX_PP_STRIP_PARENS(F)>::type>()               \
        {                                                                     \
            /*If you encounter this assert while compiling code, that means   \
            that you have a HPX_UTIL_REGISTER_[UNIQUE_]FUNCTION macro         \
            somewhere in a source file, but the header in which the function  \
            is defined misses a HPX_UTIL_REGISTER_[UNIQUE_]FUNCTION_DECLARATION*/\
            static_assert(                                                    \
                get_function_name_declared<                                   \
                    VTable, std::decay<HPX_PP_STRIP_PARENS(F)>::type>::value, \
                "HPX_UTIL_REGISTER_[UNIQUE_]FUNCTION_DECLARATION missing for "\
                HPX_PP_STRINGIZE(Name));                                      \
            return HPX_PP_STRINGIZE(Name);                                    \
        }                                                                     \
    }}}                                                                       \
/**/

#endif
