//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_SERIALIZE_AS_FUTURE_AUG_08_2014_0853PM)
#define HPX_TRAITS_SERIALIZE_AS_FUTURE_AUG_08_2014_0853PM

#include <hpx/lcos/wait_all.hpp>
#include <hpx/traits.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/util/detail/pack.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/fusion/include/for_each.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename Enable>
    struct serialize_as_future
      : boost::mpl::false_
    {
        static void call(Future& f) {}
    };

    template <typename T>
    struct serialize_as_future<T const>
      : serialize_as_future<T>
    {};

    template <typename T>
    struct serialize_as_future<T&>
      : serialize_as_future<T>
    {};

    template <typename T>
    struct serialize_as_future<T&&>
      : serialize_as_future<T>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    struct serialize_as_future<Future
        , typename boost::enable_if<is_future<Future> >::type>
      : boost::mpl::true_
    {
        static void call(Future& f)
        {
            hpx::lcos::wait_all(f);
        }
    };

    template <typename Range>
    struct serialize_as_future<Range
        , typename boost::enable_if<is_future_range<Range> >::type>
      : boost::mpl::true_
    {
        static void call(Range& r)
        {
            hpx::lcos::wait_all(r);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct serialize_as_future_helper
        {
            template <typename T>
            void operator()(T& t) const
            {
                serialize_as_future<T>::call(t);
            }
        };
    }

    template <typename ...Ts>
    struct serialize_as_future<util::tuple<Ts...> >
      : util::detail::any_of<serialize_as_future<Ts>...>
    {
        static void call(util::tuple<Ts...>& t)
        {
            boost::fusion::for_each(t, detail::serialize_as_future_helper());
        }
    };
}}

#endif
