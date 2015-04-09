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
    namespace functional
    {
        struct segmented_iterator_segment
        {
            template <typename Iterator>
            struct apply
            {
                typedef typename segmented_iterator_traits<
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
                    return segmented_iterator_traits<Iterator>::segment(iter);
                };
            };
        };

        struct segmented_iterator_local
        {
            template <typename Iterator>
            struct apply
            {
                typedef typename segmented_iterator_traits<
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
                    return segmented_iterator_traits<Iterator>::local(iter);
                };
            };
        };

        struct segmented_iterator_begin
        {
            template <typename Iterator>
            struct apply
            {
                typedef typename segmented_iterator_traits<
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
                    return segmented_iterator_traits<Iterator>::begin(iter);
                };
            };
        };

        struct segmented_iterator_end
        {
            template <typename Iterator>
            struct apply
            {
                typedef typename segmented_iterator_traits<
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
                    return segmented_iterator_traits<Iterator>::end(iter);
                };
            };
        };

        struct segmented_iterator_local_begin
        {
            template <typename Iterator>
            struct apply
            {
                typedef typename segmented_iterator_traits<
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
                    return segmented_iterator_traits<Iterator>::begin(iter);
                };
            };
        };

        struct segmented_iterator_local_end
        {
            template <typename Iterator>
            struct apply
            {
                typedef typename segmented_iterator_traits<
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
                    return segmented_iterator_traits<Iterator>::end(iter);
                };
            };
        };
    }
}}

#endif

