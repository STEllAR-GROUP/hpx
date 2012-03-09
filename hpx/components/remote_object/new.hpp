//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_COMPONENTS_REMOTE_OBJECT_NEW_HPP
#define HPX_COMPONENTS_REMOTE_OBJECT_NEW_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/components/remote_object/stubs/remote_object.hpp>
#include <hpx/components/remote_object/ctor_fun.hpp>
#include <hpx/components/remote_object/object.hpp>
#include <hpx/components/remote_object/new_impl.hpp>

namespace hpx { namespace components
{
    namespace remote_object
    {
        template <typename T>
        struct dtor_fun
        {
            typedef void result_type;

            void operator()(void ** o) const
            {
                delete (*reinterpret_cast<T **>(o));
            }

            template <typename Archive>
            void serialize(Archive &, unsigned)
            {}
        };
    }
    // asynchronously creates an instance of type T on locality target_id
    template <typename T>
    lcos::future<object<T> >
    new_(naming::id_type const & target_id)
    {
        return
            lcos::eager_future<
                remote_object::new_impl_action
              , object<T>
            >(
                target_id
              , target_id
              , remote_object::ctor_fun<T>()
              , remote_object::dtor_fun<T>()
            ).get_future();
    }

    // vertical repetition code to enable constructor parameters up to
    // HPX_FUNCTION limit
#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            1                                                                   \
          , HPX_FUNCTION_LIMIT                                                  \
          , <hpx/components/remote_object/new.hpp>                              \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

}}

#endif

#else

#define N BOOST_PP_ITERATION()

    template <typename T, BOOST_PP_ENUM_PARAMS(N, typename A)>
    lcos::future<object<T> >
    new_(naming::id_type const & target_id, BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a))
    {
        return
            lcos::eager_future<
                remote_object::new_impl_action
              , object<T>
            >(
                target_id
              , target_id
              , remote_object::ctor_fun<
                    T
                  , BOOST_PP_ENUM_PARAMS(N, A)
                >(
                    BOOST_PP_ENUM_PARAMS(N, a)
                )
              , remote_object::dtor_fun<T>()
            ).get_future();
    }

#undef N
#endif
