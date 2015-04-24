//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_SERIALIZE_AS_FUTURE_AUG_08_2014_0853PM)
#define HPX_TRAITS_SERIALIZE_AS_FUTURE_AUG_08_2014_0853PM

#include <hpx/config/forceinline.hpp>
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
    // This trait is used by invoke_when_ready to decide whether a parcel can
    // be sent directly (no futures are being wrapped by the continuation or
    // argument data structures), or if the parcel has to be delayed until all
    // futures have become ready.
    //
    // If the trait statically evaluates to true_, the parcel will be delayed
    // in any case and call() will be invoked on this trait to wait for all
    // (embedded) futures to become ready.
    // If it statically evaluates to false_, the decision whether the parcel
    // will be delayed is done at runtime. The function call_if() will be
    // invoked in this case to decide whether the parcel has to be delayed or
    // not. If call_if() returns true the parcel will be delayed and call()
    // will be invoked to wait for the (embedded) futures to become ready.
    //
    template <typename Future, typename Enable>
    struct serialize_as_future
      : boost::mpl::false_
    {
        static BOOST_FORCEINLINE bool call_if(Future&) { return false; }
        static BOOST_FORCEINLINE void call(Future&) {}
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
        static BOOST_FORCEINLINE bool call_if(Range& r) { return true; }

        static void call(Range& r)
        {
            hpx::lcos::wait_all(r);
        }
    };
}}

#endif
