//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SEGMENTED_ITERATOR_TRAITS_OCT_14_2014_0229PM)
#define HPX_SEGMENTED_ITERATOR_TRAITS_OCT_14_2014_0229PM

#include <hpx/traits.hpp>

#include <boost/mpl/bool.hpp>

namespace hpx { namespace traits
{
    template <typename Iterator, typename Enable>
    struct segmented_iterator_traits
    {
        typedef boost::mpl::false_ is_segmented_iterator;
    };
}}

#endif

