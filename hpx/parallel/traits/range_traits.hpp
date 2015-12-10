//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_RANGE_TRAITS_JUL_18_2015_1107AM)
#define HPX_PARALLEL_TRAITS_RANGE_TRAITS_JUL_18_2015_1107AM

#include <hpx/config.hpp>
#include <hpx/util/decay.hpp>

#include <hpx/parallel/traits/is_range.hpp>

#include <boost/range/iterator_range_core.hpp>

#include <type_traits>

namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Rng, typename Enable = void>
        struct range_traits
        {};

        template <typename Cont>
        struct range_traits<Cont,
            typename std::enable_if<
                traits::detail::is_container<Cont>::value &&
               !traits::detail::is_iterator_range<Cont>::value
            >::type>
        {
            typedef typename Cont::iterator iterator_type;
            typedef typename Cont::iterator sentinel_type;
        };

        template <typename Iterator>
        struct range_traits<boost::iterator_range<Iterator> >
        {
            typedef Iterator iterator_type;
            typedef Iterator sentinel_type;
        };
    }

    template <typename Rng, typename Enable = void>
    struct range_traits
      : detail::range_traits<typename hpx::util::decay<Rng>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Rng, typename Enable = void>
        struct range_iterator
        {};

        template <typename Rng>
        struct range_iterator<Rng,
            typename std::enable_if<traits::is_range<Rng>::value>::type>
        {
            typedef typename range_traits<Rng>::iterator_type type;
        };
    }

    template <typename Rng, typename Enable = void>
    struct range_iterator
      : detail::range_iterator<typename hpx::util::decay<Rng>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Rng, typename Enable = void>
    struct range_sentinel
    {};

    template <typename Rng>
    struct range_sentinel<Rng,
        typename std::enable_if<traits::is_range<Rng>::value>::type>
    {
        typedef typename range_traits<Rng>::sentinel_type type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Rng, typename Enable = void>
    struct range_value_type
    {};

    template <typename Rng>
    struct range_value_type<Rng,
        typename std::enable_if<traits::is_range<Rng>::value>::type>
    {
        typedef typename range_traits<Rng>::iterator_type iterator_type;
        typedef typename std::iterator_traits<iterator_type>::value_type type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Rng, typename Enable = void>
    struct range_reference
    {};

    template <typename Rng>
    struct range_reference<Rng,
        typename std::enable_if<traits::is_range<Rng>::value>::type>
    {
        typedef typename range_traits<Rng>::iterator_type iterator_type;
        typedef typename std::iterator_traits<iterator_type>::reference type;
    };
}}}

#endif
