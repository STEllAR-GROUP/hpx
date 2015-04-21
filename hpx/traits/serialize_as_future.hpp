//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_SERIALIZE_AS_FUTURE_AUG_08_2014_0853PM)
#define HPX_TRAITS_SERIALIZE_AS_FUTURE_AUG_08_2014_0853PM

#include <hpx/traits.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/utility/enable_if.hpp>

#include <vector>

namespace hpx { namespace lcos
{
    // forward declaration only
    template <typename Future>
    void wait_all(std::vector<Future> const& values);

    template <typename Future>
    void wait_all(std::vector<Future>& values);

    template <typename Future>
    void wait_all(std::vector<Future>&& values);
}}

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
}}

#endif
