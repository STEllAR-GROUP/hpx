//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SEGMENTED_ITERATOR_TRAITS_OCT_14_2014_0229PM)
#define HPX_SEGMENTED_ITERATOR_TRAITS_OCT_14_2014_0229PM

#include <hpx/traits.hpp>
#include <type_traits>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator, typename Enable>
    struct segmented_iterator_traits
    {
        typedef std::false_type is_segmented_iterator;
    };

    template <typename Iterator, typename Enable>
    struct is_segmented_iterator
      : segmented_iterator_traits<Iterator>::is_segmented_iterator
    {};

    ///////////////////////////////////////////////////////////////////////////
    // traits allowing to distinguish iterators which have a purely local
    // representation
    template <typename Iterator, typename Enable>
    struct segmented_local_iterator_traits
    {
        typedef std::false_type is_segmented_local_iterator;

        typedef Iterator iterator;
        typedef Iterator local_iterator;
        typedef Iterator local_raw_iterator;

        static local_raw_iterator const& local(local_iterator const& it)
        {
            return it;
        }

        static local_iterator const& remote(local_raw_iterator const& it)
        {
            return it;
        }

        static local_raw_iterator local(local_iterator&& it)
        {
            return std::move(it);
        }

        static local_iterator remote(local_raw_iterator&& it)
        {
            return std::move(it);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable>
    struct projected_iterator
    {
        typedef typename hpx::util::decay<T>::type type;
    };
}}

#endif

