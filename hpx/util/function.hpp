//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_FUNCTION_HPP
#define HPX_UTIL_FUNCTION_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/error.hpp>
#include <hpx/runtime/actions/guid_initialization.hpp>
#include <hpx/util/detail/function_template.hpp>
#include <hpx/util/detail/vtable_ptr_base.hpp>
#include <hpx/util/detail/vtable_ptr.hpp>
#include <hpx/util/detail/get_table.hpp>
#include <hpx/util/detail/vtable.hpp>
#include <hpx/util/detail/get_empty_table.hpp>
#include <hpx/util/detail/empty_vtable.hpp>
#include <hpx/util/decay.hpp>

#include <boost/preprocessor/cat.hpp>

///////////////////////////////////////////////////////////////////////////////
#define HPX_CONTINUATION_REGISTER_FUNCTION_FACTORY(Vtable, Name)              \
    static ::hpx::util::detail::function_registration<Vtable>                 \
        const BOOST_PP_CAT(Name, _function_factory_registration) =            \
            ::hpx::util::detail::function_registration<Vtable>();             \
/**/

#define HPX_DECLARE_GET_FUNCTION_NAME(Vtable, Name)                           \
    namespace hpx { namespace util { namespace detail {                       \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_function_name<Vtable>();                              \
    }}}                                                                       \
/**/

#define HPX_UTIL_REGISTER_FUNCTION_DECLARATION(Sig, Functor, Name)            \
    namespace hpx { namespace util { namespace detail {                       \
        typedef                                                               \
            vtable_ptr<                                                       \
                Sig                                                           \
              , portable_binary_iarchive                                      \
              , portable_binary_oarchive                                      \
              , vtable<sizeof(util::decay<Functor>::type) <=                  \
                       sizeof(void *)>::type<                                 \
                    util::decay<Functor>::type                                \
                  , Sig                                                       \
                  , portable_binary_iarchive                                  \
                  , portable_binary_oarchive                                  \
                >                                                             \
            >                                                                 \
            BOOST_PP_CAT(BOOST_PP_CAT(__,                                     \
                BOOST_PP_CAT(hpx_function_serialization_, Name)), _type);     \
    }}}                                                                       \
    HPX_DECLARE_GET_FUNCTION_NAME(                                            \
        BOOST_PP_CAT(BOOST_PP_CAT(hpx::util::detail::__,                      \
            BOOST_PP_CAT(hpx_function_serialization_, Name)), _type)          \
        , Name)                                                               \
    namespace hpx { namespace traits {                                        \
        template <>                                                           \
        struct needs_automatic_registration<                                  \
            BOOST_PP_CAT(BOOST_PP_CAT(util::detail::__,                       \
                BOOST_PP_CAT(hpx_function_serialization_, Name)), _type)>     \
          : boost::mpl::false_                                                \
        {};                                                                   \
    }}                                                                        \
/**/

#define HPX_DEFINE_GET_FUNCTION_NAME(Vtable, Name)                            \
    namespace hpx { namespace util { namespace detail {                       \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_function_name<Vtable>()                               \
        {                                                                     \
            return BOOST_PP_STRINGIZE(name);                                  \
        }                                                                     \
    }}}                                                                       \
/**/

#define HPX_UTIL_REGISTER_FUNCTION(Sig, Functor, Name)                        \
    HPX_CONTINUATION_REGISTER_FUNCTION_FACTORY(                               \
        BOOST_PP_CAT(BOOST_PP_CAT(hpx::util::detail::__,                      \
            BOOST_PP_CAT(hpx_function_serialization_, Name)), _type)          \
      , Name)                                                                 \
    HPX_DEFINE_GET_FUNCTION_NAME(                                             \
        BOOST_PP_CAT(BOOST_PP_CAT(hpx::util::detail::__,                      \
            BOOST_PP_CAT(hpx_function_serialization_, Name)), _type)          \
      , Name)                                                                 \
/**/

#endif
