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
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/preprocessor/stringize.hpp>

#include <type_traits>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable, typename T>
    struct get_function_name_declared
      : std::false_type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable, typename Functor>
    struct get_function_name_impl
    {
        static char const* call()
#ifdef HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION
        {
            return util::type_id<Functor>::typeid_.type_id();
        }
#else
        = delete;
#endif
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable, typename Functor>
    char const* get_function_name()
    {
        return get_function_name_impl<VTable, Functor>::call();
    }
}}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_DECLARE_GET_FUNCTION_NAME(VTable, Functor, Name)                  \
    namespace hpx { namespace util { namespace detail {                       \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_function_name<                                        \
            VTable, std::decay<HPX_UTIL_STRIP(Functor)>::type>();             \
                                                                              \
        template <>                                                           \
        struct get_function_name_declared<                                    \
            VTable, std::decay<HPX_UTIL_STRIP(Functor)>::type                 \
        > : std::true_type                                                    \
        {};                                                                   \
    }}}                                                                       \
/**/

#define HPX_DEFINE_GET_FUNCTION_NAME(VTable, Functor, Name)                   \
    namespace hpx { namespace util { namespace detail {                       \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_function_name<                                        \
            VTable, std::decay<HPX_UTIL_STRIP(Functor)>::type>()              \
        {                                                                     \
            /*If you encounter this assert while compiling code, that means   \
            that you have a HPX_UTIL_REGISTER_[UNIQUE_]FUNCTION macro         \
            somewhere in a source file, but the header in which the function  \
            is defined misses a HPX_UTIL_REGISTER_[UNIQUE_]FUNCTION_DECLARATION*/\
            static_assert(                                                    \
                get_function_name_declared<                                   \
                    VTable, std::decay<HPX_UTIL_STRIP(Functor)>::type>::value,\
                "HPX_UTIL_REGISTER_[UNIQUE_]FUNCTION_DECLARATION missing for "\
                BOOST_PP_STRINGIZE(Name));                                    \
            return BOOST_PP_STRINGIZE(Name);                                  \
        }                                                                     \
    }}}                                                                       \
/**/

#endif
