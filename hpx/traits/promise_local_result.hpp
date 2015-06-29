//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_PROMISE_LOCAL_RESULT_OCT_27_2011_0416PM)
#define HPX_TRAITS_PROMISE_LOCAL_RESULT_OCT_27_2011_0416PM

#include <hpx/config.hpp>
#include <hpx/traits.hpp>
#include <hpx/util/unused.hpp>

#include <boost/mpl/identity.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename Enable>
    struct promise_local_result
      : boost::mpl::identity<Result>
    {};

    template <>
    struct promise_local_result<util::unused_type>
      : boost::mpl::identity<void>
    {};
}}

#endif
