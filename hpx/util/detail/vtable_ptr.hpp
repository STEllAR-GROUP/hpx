//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_FUNCTION_DETAIL_VTABLE_PTR_HPP
#define HPX_FUNCTION_DETAIL_VTABLE_PTR_HPP

#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <hpx/util/detail/vtable_ptr_fwd.hpp>
#include <hpx/util/detail/vtable_ptr_base_fwd.hpp>
#include <hpx/util/detail/serialization_registration.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            0                                                                   \
          , HPX_FUNCTION_LIMIT                                                  \
          , <hpx/util/detail/vtable_ptr.hpp>                                    \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (
        template <
            typename Sig
          , typename IArchive
          , typename OArchive
          , typename Vtable
        >
    )
  , (
        hpx::util::detail::vtable_ptr<Sig, IArchive, OArchive, Vtable>
    )
)

#endif

#else

#define N BOOST_PP_ITERATION()

namespace hpx { namespace util { namespace detail {

    template <
        typename R
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct vtable_ptr<
        R(BOOST_PP_ENUM_PARAMS(N, A))
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(BOOST_PP_ENUM_PARAMS(N, A))
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(BOOST_PP_ENUM_PARAMS(N, A))
              , IArchive
              , OArchive
            >
            base_type;

        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::move = Vtable::move;
            base_type::invoke = Vtable::invoke;
            base_type::iserialize = Vtable::iserialize;
            base_type::oserialize = Vtable::oserialize;
        }

        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<vtable_ptr, base_type>();
        }

        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }


        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
}}}


#endif

