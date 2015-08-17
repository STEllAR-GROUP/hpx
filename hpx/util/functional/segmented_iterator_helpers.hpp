//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_FUNCTIONAL_SEGMENTED_ITERATOR_HELPERS_JUL_20_2015_1105AM)
#define HPX_UTIL_FUNCTIONAL_SEGMENTED_ITERATOR_HELPERS_JUL_20_2015_1105AM

#include <hpx/traits.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

namespace hpx { namespace util { namespace functional
{
    ///////////////////////////////////////////////////////////////////////////
    struct segmented_iterator_segment
    {
        template <typename Iterator>
        struct apply
        {
            typedef typename traits::segmented_iterator_traits<
                    Iterator
                >::segment_iterator type;

            template <typename T>
            struct result;

            template <typename This, typename Iter>
            struct result<This(Iter)>
            {
                typedef typename apply::type type;
            };

            template <typename Iter>
            typename result<segmented_iterator_segment(Iter)>::type
            operator()(Iter iter) const
            {
                return traits::segmented_iterator_traits<Iterator>::segment(iter);
            };
        };
    };

    struct segmented_iterator_local
    {
        template <typename Iterator>
        struct apply
        {
            typedef typename traits::segmented_iterator_traits<
                    Iterator
                >::local_iterator type;

            template <typename T>
            struct result;

            template <typename This, typename Iter>
            struct result<This(Iter)>
            {
                typedef typename apply::type type;
            };

            template <typename Iter>
            typename result<segmented_iterator_local(Iter)>::type
            operator()(Iter iter) const
            {
                return traits::segmented_iterator_traits<Iterator>::local(iter);
            };
        };
    };

    struct segmented_iterator_begin
    {
        template <typename Iterator>
        struct apply
        {
            typedef typename traits::segmented_iterator_traits<
                    Iterator
                >::local_iterator type;

            template <typename T>
            struct result;

            template <typename This, typename SegIter>
            struct result<This(SegIter)>
            {
                typedef typename apply::type type;
            };

            template <typename SegIter>
            typename result<segmented_iterator_begin(SegIter)>::type
            operator()(SegIter iter) const
            {
                return traits::segmented_iterator_traits<Iterator>::begin(iter);
            };
        };
    };

    struct segmented_iterator_end
    {
        template <typename Iterator>
        struct apply
        {
            typedef typename traits::segmented_iterator_traits<
                    Iterator
                >::local_iterator type;

            template <typename T>
            struct result;

            template <typename This, typename SegIter>
            struct result<This(SegIter)>
            {
                typedef typename apply::type type;
            };

            template <typename SegIter>
            typename result<segmented_iterator_end(SegIter)>::type
            operator()(SegIter iter) const
            {
                return traits::segmented_iterator_traits<Iterator>::end(iter);
            };
        };
    };

    struct segmented_iterator_local_begin
    {
        template <typename Iterator>
        struct apply
        {
            typedef typename traits::segmented_iterator_traits<
                    Iterator
                >::local_raw_iterator type;

            template <typename T>
            struct result;

            template <typename This, typename LocalSegIter>
            struct result<This(LocalSegIter)>
            {
                typedef typename apply::type type;
            };

            template <typename LocalSegIter>
            typename result<segmented_iterator_local_begin(LocalSegIter)>::type
            operator()(LocalSegIter iter) const
            {
                return traits::segmented_iterator_traits<Iterator>::begin(iter);
            };
        };
    };

    struct segmented_iterator_local_end
    {
        template <typename Iterator>
        struct apply
        {
            typedef typename traits::segmented_iterator_traits<
                    Iterator
                >::local_raw_iterator type;

            template <typename T>
            struct result;

            template <typename This, typename LocalSegIter>
            struct result<This(LocalSegIter)>
            {
                typedef typename apply::type type;
            };

            template <typename LocalSegIter>
            typename result<segmented_iterator_local_end(LocalSegIter)>::type
            operator()(LocalSegIter iter) const
            {
                return traits::segmented_iterator_traits<Iterator>::end(iter);
            };
        };
    };
}}}

#endif

