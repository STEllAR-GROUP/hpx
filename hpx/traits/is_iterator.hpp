//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_IS_ITERATOR_MAR_05_2016_0840PM)
#define HPX_PARALLEL_TRAITS_IS_ITERATOR_MAR_05_2016_0840PM

#include <hpx/config.hpp>
#include <hpx/traits.hpp>

#include <type_traits>
#include <iterator>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // This implementation of is_iterator seems to work fine even for
        // VS2013 which has an implementation of std::iterator_traits which is
        // SFINAE-unfriendly.
        template <typename T>
        struct is_iterator
        {
            template <typename U, typename =
                typename std::iterator_traits<U>::pointer>
            static char test(U&&);

            static long test(...);

            static bool const value =
                sizeof(test(std::declval<T>())) == sizeof(char);
        };
    }

    template <typename Iter, typename Enable>
    struct is_iterator
      : detail::is_iterator<typename std::decay<Iter>::type>
    {};

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Iter, typename Cat, typename Enable = void>
        struct has_category
          : std::false_type
        {};

        template <typename Iter, typename Cat>
        struct has_category<Iter, Cat,
                typename std::enable_if<is_iterator<Iter>::value>::type>
          : std::is_same<
                typename std::iterator_traits<Iter>::iterator_category, Cat>
        {};
    }

    template <typename Iter, typename Enable>
    struct is_input_iterator
      : detail::has_category<
            typename std::decay<Iter>::type, std::input_iterator_tag>
    {};

    template <typename Iter, typename Enable>
    struct is_forward_iterator
      : detail::has_category<
            typename std::decay<Iter>::type, std::forward_iterator_tag>
    {};

    template <typename Iter, typename Enable>
    struct is_bidirectional_iterator
      : detail::has_category<
            typename std::decay<Iter>::type, std::bidirectional_iterator_tag>
    {};

    template <typename Iter, typename Enable>
    struct is_random_access_iterator
      : detail::has_category<
            typename std::decay<Iter>::type, std::random_access_iterator_tag>
    {};
}}

#endif

