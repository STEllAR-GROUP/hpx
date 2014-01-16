//  Copyright (c) 2012 Thomas Heller
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DECAY_HPP
#define HPX_UTIL_DECAY_HPP

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/config.hpp>
#include <boost/type_traits/is_array.hpp>
#include <boost/type_traits/is_function.hpp>
#include <boost/type_traits/remove_bounds.hpp>
#include <boost/type_traits/add_pointer.hpp>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/remove_reference.hpp>

namespace hpx { namespace util
{
    template <typename T>
    struct decay
    {
        typedef typename boost::remove_reference<T>::type Ty;

        typedef
            typename boost::mpl::eval_if<
                boost::is_array<Ty>
              , boost::mpl::identity<typename boost::remove_bounds<Ty>::type *>
              , typename boost::mpl::eval_if<
                    boost::is_function<Ty>
                  , boost::add_pointer<Ty>
                  , boost::remove_cv<Ty>
                >
            >::type
            type;
    };

    namespace detail
    {
        template <typename T, typename TD = typename decay<T>::type>
        struct decay_unwrap
        {
            typedef TD type;
        };

        template <typename T, typename U>
        struct decay_unwrap<T, boost::reference_wrapper<U> >
        {
            typedef U& type;
        };
    }
}}

#endif
