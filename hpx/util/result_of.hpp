//  Copyright (c) 2013-2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_RESULT_OF_HPP
#define HPX_UTIL_RESULT_OF_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/detail/qualify_as.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/or.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/is_function.hpp>
#include <boost/type_traits/is_member_function_pointer.hpp>
#include <boost/type_traits/is_member_object_pointer.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/utility/result_of.hpp>

namespace hpx { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Q = int, typename Enable = void>
        struct get_member_pointer_object
        {
            typedef
                typename detail::qualify_as<
                    typename get_member_pointer_object<T>::type
                  , typename boost::mpl::if_<
                        boost::is_pointer<typename util::decay<Q>::type>
                      , typename boost::remove_pointer<
                            typename util::decay<Q>::type>::type&
                      , Q
                    >::type
                >::type
                type;
        };

        template <typename T, typename Q>
        struct get_member_pointer_object<T, Q
          , typename boost::enable_if<
                boost::is_reference_wrapper<typename util::decay<Q>::type> >::type
        > : get_member_pointer_object<T
              , typename util::decay_unwrap<Q>::type&>
        {};

        template <typename T, typename C>
        struct get_member_pointer_object<T C::*>
        {
            typedef T type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename FD, typename F, typename Enable = void>
        struct result_of_impl
          : boost::result_of<F>
        {};

        template <typename FD, typename F, typename ...Ts>
        struct result_of_impl<FD, F(Ts...)
          , typename boost::enable_if<boost::is_reference_wrapper<FD> >::type
        > : boost::result_of<typename boost::unwrap_reference<FD>::type&(Ts...)>
        {};

        /* workaround for tricking result_of into using decltype */
        template <typename FD, typename F, typename ...Ts>
        struct result_of_impl<FD, F(Ts...)
          , typename boost::enable_if<
                boost::mpl::or_<
                    boost::is_function<typename boost::remove_pointer<FD>::type>
                  , boost::is_member_function_pointer<FD>
                >
            >::type
        > : boost::result_of<FD(Ts...)>
        {};

        template <typename FD, typename F, typename Class>
        struct result_of_impl<FD, F(Class)
          , typename boost::enable_if<boost::is_member_object_pointer<FD> >::type
        > : detail::get_member_pointer_object<FD, Class>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    struct result_of;

    template <typename F, typename ...Ts>
    struct result_of<F(Ts...)>
      : detail::result_of_impl<typename hpx::util::decay<F>::type, F(Ts...)>
    {};

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Fallback>
        struct result_of_or
          : boost::mpl::eval_if_c<
                traits::is_callable<T>::value
              , hpx::util::result_of<T>
              , boost::mpl::identity<Fallback>
            >
        {};
    }
}}

#endif
