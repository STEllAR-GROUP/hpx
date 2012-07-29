//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_FUNCTION_DETAIL_VTABLE_PTR_BASE_HPP
#define HPX_FUNCTION_DETAIL_VTABLE_PTR_BASE_HPP

#include <boost/detail/sp_typeinfo.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <hpx/util/detail/vtable_ptr_base_fwd.hpp>

#if !defined(HPX_DONT_USE_PREPROCESSED_FILES)
#  include <hpx/util/detail/preprocessed/vtable_ptr_base.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/vtable_ptr_base_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            0                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                         \
          , <hpx/util/detail/vtable_ptr_base.hpp>                               \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_DONT_USE_PREPROCESSED_FILES)

#endif

#else

#define N BOOST_PP_ITERATION()

namespace hpx { namespace util { namespace detail {

    template <
        typename R
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(BOOST_PP_ENUM_PARAMS(N, A))
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}

        virtual vtable_ptr_base * get_ptr() = 0;

        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, A));

        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };

    template <
        typename R
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    struct vtable_ptr_base<
        R(BOOST_PP_ENUM_PARAMS(N, A))
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}

        virtual vtable_ptr_base * get_ptr() = 0;

        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, A));
    };
}}}

#undef N
#endif
