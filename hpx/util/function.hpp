//  Copyright (c) 2011 Thomas Heller
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

#define HPX_UTIL_REGISTER_FUNCTION_DECLARATION(Sig, Functor, Name)            \
    namespace hpx { namespace traits {                                        \
        template <>                                                           \
        struct needs_guid_initialization<                                     \
            hpx::util::detail::vtable_ptr<                                    \
                Sig                                                           \
              , hpx::util::portable_binary_iarchive                           \
              , hpx::util::portable_binary_oarchive                           \
              , hpx::util::detail::vtable<sizeof(Functor) <= sizeof(void *)>::type<\
                    Functor                                                   \
                  , Sig                                                       \
                  , hpx::util::portable_binary_iarchive                       \
                  , hpx::util::portable_binary_oarchive                       \
                >                                                             \
            >                                                                 \
        >                                                                     \
          : boost::mpl::false_                                                \
        {};                                                                   \
    }}                                                                        \
    namespace boost { namespace archive { namespace detail {                  \
        namespace extra_detail {                                              \
            template <>                                                       \
            struct init_guid<                                                 \
                hpx::util::detail::vtable_ptr<                                \
                    Sig                                                       \
                  , hpx::util::portable_binary_iarchive                       \
                  , hpx::util::portable_binary_oarchive                       \
                  , hpx::util::detail::vtable<sizeof(Functor) <= sizeof(void *)>::type<\
                        Functor                                               \
                      , Sig                                                   \
                      , hpx::util::portable_binary_iarchive                   \
                      , hpx::util::portable_binary_oarchive                   \
                    >                                                         \
                >                                                             \
            >;                                                                \
        }                                                                     \
    }}}                                                                       \
    typedef                                                                   \
        hpx::util::detail::vtable_ptr<                                        \
            Sig                                                               \
          , hpx::util::portable_binary_iarchive                               \
          , hpx::util::portable_binary_oarchive                               \
          , hpx::util::detail::vtable<sizeof(Functor) <= sizeof(void *)>::type<\
                Functor                                                       \
              , Sig                                                           \
              , hpx::util::portable_binary_iarchive                           \
              , hpx::util::portable_binary_oarchive                           \
            >                                                                 \
        >                                                                     \
        BOOST_PP_CAT(BOOST_PP_CAT(__, BOOST_PP_CAT(hpx_function_serialization_, Name)), _type); \
    BOOST_CLASS_EXPORT_KEY2(                                                  \
        BOOST_PP_CAT(BOOST_PP_CAT(__, BOOST_PP_CAT(hpx_function_serialization_, Name)), _type),\
        BOOST_PP_STRINGIZE(Name))                                             \
/**/
#define HPX_UTIL_REGISTER_FUNCTION(Sig, Functor, Name)                        \
    HPX_REGISTER_BASE_HELPER(                                                 \
        BOOST_PP_CAT(                                                         \
            BOOST_PP_CAT(                                                     \
                __                                                            \
              , BOOST_PP_CAT(                                                 \
                    hpx_function_serialization_                               \
                  , Name                                                      \
                )                                                             \
            )                                                                 \
          , _type                                                             \
        ), Name)                                                              \
    BOOST_CLASS_EXPORT_IMPLEMENT(                                             \
        BOOST_PP_CAT(                                                         \
            BOOST_PP_CAT(                                                     \
                __                                                            \
              , BOOST_PP_CAT(                                                 \
                    hpx_function_serialization_                               \
                  , Name                                                      \
                )                                                             \
            )                                                                 \
          , _type                                                             \
        )                                                                     \
    )                                                                         \
/**/

#endif
