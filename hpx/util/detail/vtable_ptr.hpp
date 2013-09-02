//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_FUNCTION_DETAIL_VTABLE_PTR_HPP
#define HPX_FUNCTION_DETAIL_VTABLE_PTR_HPP

#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/mpl/int.hpp>

#include <hpx/runtime/actions/guid_initialization.hpp>
#include <hpx/util/detail/vtable_ptr_fwd.hpp>
#include <hpx/util/detail/vtable_ptr_base_fwd.hpp>
#include <hpx/util/detail/serialization_registration.hpp>
#include <hpx/util/void_cast.hpp>

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/detail/preprocessed/vtable_ptr.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/vtable_ptr_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            0                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                         \
          , <hpx/util/detail/vtable_ptr.hpp>                                    \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

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

    ///////////////////////////////////////////////////////////////////////////
    template <typename Function>
    struct init_registration;

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
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }

        ~vtable_ptr()
        {
            init_registration<vtable_ptr>::g.register_function();
        }

        char const* get_function_name() const
        {
            return get_function_name<vtable_ptr>();
        }

        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }

        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar << Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar >> Vtable::construct(object);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // registration code for serialization
    template <
        typename R
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct init_registration<
        vtable_ptr<
            R(BOOST_PP_ENUM_PARAMS(N, A))
          , IArchive
          , OArchive
          , Vtable
        >
    >
    {
        typedef vtable_ptr<
            R(BOOST_PP_ENUM_PARAMS(N, A))
          , IArchive
          , OArchive
          , Vtable
        > vtable_ptr_type;

        static automatic_function_registration<vtable_ptr_type> g;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename R
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
      , typename Vtable
    >
    struct vtable_ptr<
        R(BOOST_PP_ENUM_PARAMS(N, A))
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(BOOST_PP_ENUM_PARAMS(N, A))
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(BOOST_PP_ENUM_PARAMS(N, A))
              , void
              , void
            >
            base_type;

        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}

namespace boost { namespace serialization {

    template <
        typename R
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(BOOST_PP_ENUM_PARAMS(N, A)), IArchive, OArchive, Vtable
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};

}}

#undef N

#endif

