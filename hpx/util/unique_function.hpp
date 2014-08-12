//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_UNIQUE_FUNCTION_HPP
#define HPX_UTIL_UNIQUE_FUNCTION_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/error.hpp>
#include <hpx/runtime/actions/guid_initialization.hpp>
#include <hpx/util/detail/unique_function_template.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>
#include <hpx/util/decay.hpp>

#include <boost/preprocessor/cat.hpp>

#include <utility>

///////////////////////////////////////////////////////////////////////////////
#define HPX_CONTINUATION_REGISTER_UNIQUE_FUNCTION_FACTORY(VTable, Name)       \
    static ::hpx::util::detail::function_registration<                        \
        VTable::first_type, VTable::second_type                               \
    > const BOOST_PP_CAT(Name, _unique_function_factory_registration) =       \
            ::hpx::util::detail::function_registration<                       \
                VTable::first_type, VTable::second_type                       \
            >();                                                              \
/**/

#define HPX_DECLARE_GET_UNIQUE_FUNCTION_NAME(VTable, Name)                    \
    namespace hpx { namespace util { namespace detail {                       \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_function_name<VTable>();                              \
    }}}                                                                       \
/**/

#define HPX_UTIL_REGISTER_UNIQUE_FUNCTION_DECLARATION(Sig, Functor, Name)     \
    namespace hpx { namespace util { namespace detail {                       \
        typedef                                                               \
            std::pair<                                                        \
                unique_function_vtable_ptr<                                   \
                    Sig                                                       \
                  , portable_binary_iarchive, portable_binary_oarchive        \
                >                                                             \
              , util::decay<HPX_UTIL_STRIP(Functor)>::type                    \
            >                                                                 \
            BOOST_PP_CAT(BOOST_PP_CAT(__,                                     \
                BOOST_PP_CAT(hpx_unique_function_serialization_, Name)), _type);\
    }}}                                                                       \
    HPX_DECLARE_GET_UNIQUE_FUNCTION_NAME(                                     \
        BOOST_PP_CAT(BOOST_PP_CAT(hpx::util::detail::__,                      \
            BOOST_PP_CAT(hpx_unique_function_serialization_, Name)), _type)   \
        , Name)                                                               \
    namespace hpx { namespace traits {                                        \
        template <>                                                           \
        struct needs_automatic_registration<                                  \
            BOOST_PP_CAT(BOOST_PP_CAT(util::detail::__,                       \
                BOOST_PP_CAT(hpx_unique_function_serialization_, Name)), _type)>\
          : boost::mpl::false_                                                \
        {};                                                                   \
    }}                                                                        \
/**/

#define HPX_DEFINE_GET_UNIQUE_FUNCTION_NAME(VTable, Name)                     \
    namespace hpx { namespace util { namespace detail {                       \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_function_name<VTable>()                               \
        {                                                                     \
            return BOOST_PP_STRINGIZE(Name);                                  \
        }                                                                     \
    }}}                                                                       \
/**/

#define HPX_UTIL_REGISTER_UNIQUE_FUNCTION(Sig, Functor, Name)                 \
    HPX_CONTINUATION_REGISTER_UNIQUE_FUNCTION_FACTORY(                        \
        BOOST_PP_CAT(BOOST_PP_CAT(hpx::util::detail::__,                      \
            BOOST_PP_CAT(hpx_function_serialization_, Name)), _type)          \
      , Name)                                                                 \
    HPX_DEFINE_GET_UNIQUE_FUNCTION_NAME(                                      \
        BOOST_PP_CAT(BOOST_PP_CAT(hpx::util::detail::__,                      \
            BOOST_PP_CAT(hpx_function_serialization_, Name)), _type)          \
      , Name)                                                                 \
/**/

#endif
