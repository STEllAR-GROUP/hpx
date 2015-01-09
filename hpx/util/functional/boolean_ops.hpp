//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_FUNCTIONAL_JAN_09_2015_1053AM)
#define HPX_UTIL_FUNCTIONAL_JAN_09_2015_1053AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/or.hpp>

namespace hpx { namespace util { namespace functional
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ...Ts>
    struct all_of
      : boost::mpl::fold<
            util::tuple<Ts...>, boost::mpl::true_,
            boost::mpl::and_<boost::mpl::_1, boost::mpl::_2>
        >::type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename ...Ts>
    struct any_of
      : boost::mpl::fold<
            util::tuple<Ts...>, boost::mpl::false_,
            boost::mpl::or_<boost::mpl::_1, boost::mpl::_2>
        >::type
    {};
}}}

#endif
