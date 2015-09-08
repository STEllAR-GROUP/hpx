//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_FUNCTION_HPP
#define HPX_UTIL_FUNCTION_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/error.hpp>
#include <hpx/util/detail/function_template.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/traits/needs_automatic_registration.hpp>

#include <boost/preprocessor/cat.hpp>

#include <utility>

///////////////////////////////////////////////////////////////////////////////
#define HPX_CONTINUATION_REGISTER_FUNCTION_FACTORY(VTable, Name)              \
    static ::hpx::util::detail::function_registration<                        \
        ::hpx::util::tuple_element<0, VTable>::type                           \
      , ::hpx::util::tuple_element<1, VTable>::type                           \
    > const BOOST_PP_CAT(Name, _function_factory_registration) =              \
            ::hpx::util::detail::function_registration<                       \
                ::hpx::util::tuple_element<0, VTable>::type                   \
              , ::hpx::util::tuple_element<1, VTable>::type                   \
            >();                                                              \
/**/

#define HPX_DECLARE_GET_FUNCTION_NAME(VTable, Name)                           \
    namespace hpx { namespace util { namespace detail {                       \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_function_name<VTable>();                              \
    }}}                                                                       \
/**/

#define HPX_UTIL_REGISTER_FUNCTION_DECLARATION(Sig, Functor, Name)            \
    namespace hpx { namespace util { namespace detail {                       \
        typedef                                                               \
            hpx::util::tuple<                                                 \
                function_vtable_ptr<                                          \
                    Sig                                                       \
                  , ::hpx::serialization::input_archive                       \
                  , ::hpx::serialization::output_archive                      \
                >                                                             \
              , util::decay<HPX_UTIL_STRIP(Functor)>::type                    \
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

#define HPX_DEFINE_GET_FUNCTION_NAME(VTable, Name)                            \
    namespace hpx { namespace util { namespace detail {                       \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_function_name<VTable>()                               \
        {                                                                     \
            return BOOST_PP_STRINGIZE(Name);                                  \
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


// Pseudo registration for empty functions.
// We don't want to serialize empty functions.
namespace hpx { namespace util { namespace detail
{
    template <typename Sig>
    struct get_function_name_impl<
        std::pair<
            hpx::util::detail::function_vtable_ptr<
                Sig
              , hpx::serialization::input_archive, hpx::serialization::output_archive
            >
          , hpx::util::detail::empty_function<Sig>
        >
    >
    {
        static char const * call()
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "get_function_name<empty_function>");
            return "";
        }
    };
}}}

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig>
    struct needs_automatic_registration<
        std::pair<
            hpx::util::detail::function_vtable_ptr<
                Sig
              , hpx::serialization::input_archive, hpx::serialization::output_archive
            >
          , hpx::util::detail::empty_function<Sig>
        >
    >
      : boost::mpl::false_
    {};
}}

#endif
