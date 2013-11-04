//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_DETAIL_RESULT_OF_OR_NOV_03_2013_1201PM)
#define HPX_UTIL_DETAIL_RESULT_OF_OR_NOV_03_2013_1201PM

#include <hpx/traits/is_callable.hpp>
#include <hpx/util/invoke.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Fallback>
    struct result_of_or
      : boost::mpl::eval_if_c<
            traits::is_callable<T>::value
          , util::invoke_result_of<T>
          , boost::mpl::identity<Fallback>
        >
    {};
}}}

#endif
