//  Copyright (c) 2016 Denis Demidov
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This tests verifies that #2032 remains fixed

#include <hpx/include/traits.hpp>
#include <hpx/include/lcos.hpp>

#include <boost/range/iterator_range.hpp>

#include <vector>

typedef boost::iterator_range<
        std::vector<hpx::shared_future<void> >::iterator
    > future_range;

typedef hpx::traits::is_future_range<future_range>::type error1;

typedef hpx::traits::future_range_traits<future_range>::future_type error2;

int main()
{
    return 0;
}
