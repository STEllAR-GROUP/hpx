//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_IS_ITERATOR_MAR_05_2016_0840PM)
#define HPX_PARALLEL_TRAITS_IS_ITERATOR_MAR_05_2016_0840PM

#include <hpx/config.hpp>

#include <boost/iterator/iterator_categories.hpp>

#include <iterator>
#include <type_traits>
#include <utility>

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
#if defined(HPX_MSVC) && defined(__CUDACC__)
            template <typename U>
            static typename U::iterator_category * test(U); // iterator

            template <typename U>
            static void * test(U *); // pointer
#else
            template <typename U, typename =
                typename std::iterator_traits<U>::pointer>
            static void* test(U&&);
#endif

            static char test(...);

            static bool const value =
                sizeof(test(std::declval<T>())) == sizeof(void*);
        };
    }

    template <typename Iter, typename Enable = void>
    struct is_iterator
      : detail::is_iterator<typename std::decay<Iter>::type>
    {};

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Iter, typename Cat, typename Enable = void>
        struct belongs_to_iterator_category
          : std::false_type
        {};

        template <typename Iter, typename Cat>
        struct belongs_to_iterator_category<Iter, Cat,
                typename std::enable_if<is_iterator<Iter>::value>::type>
          : std::is_base_of<
                Cat, typename std::iterator_traits<Iter>::iterator_category>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter, typename Traversal, typename Enable = void>
        struct belongs_to_iterator_traversal
          : std::false_type
        {};

        template <typename Iter, typename Traversal>
        struct belongs_to_iterator_traversal<Iter, Traversal,
                typename std::enable_if<is_iterator<Iter>::value>::type>
          : std::is_base_of<
                Traversal, typename boost::iterator_traversal<Iter>::type>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter, typename Cat, typename Enable = void>
        struct has_category
          : std::false_type
        {};

        template <typename Iter, typename Cat>
        struct has_category<Iter, Cat,
                typename std::enable_if<is_iterator<Iter>::value>::type>
          : std::is_same<
                Cat, typename std::iterator_traits<Iter>::iterator_category>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter, typename Traversal, typename Enable = void>
        struct has_traversal
          : std::false_type
        {};

        template <typename Iter, typename Traversal>
        struct has_traversal<Iter, Traversal,
                typename std::enable_if<is_iterator<Iter>::value>::type>
          : std::is_same<
                Traversal, typename boost::iterator_traversal<Iter>::type>
        {};
    }

    template <typename Iter, typename Enable = void>
    struct is_output_iterator
      : std::integral_constant<bool,
            detail::has_category<
                typename std::decay<Iter>::type, std::output_iterator_tag
            >::value
         || detail::has_traversal<
                typename std::decay<Iter>::type, boost::incrementable_traversal_tag
            >::value>
    {};

    template <typename Iter, typename Enable = void>
    struct is_input_iterator
      : std::integral_constant<bool,
            detail::belongs_to_iterator_category<
                typename std::decay<Iter>::type, std::input_iterator_tag
            >::value
         || detail::belongs_to_iterator_traversal<
                typename std::decay<Iter>::type, boost::single_pass_traversal_tag
            >::value>
    {};

    template <typename Iter, typename Enable = void>
    struct is_forward_iterator
      : std::integral_constant<bool,
            detail::belongs_to_iterator_category<
                typename std::decay<Iter>::type, std::forward_iterator_tag
            >::value
         || detail::belongs_to_iterator_traversal<
                typename std::decay<Iter>::type, boost::forward_traversal_tag
            >::value>
    {};

    template <typename Iter, typename Enable = void>
    struct is_bidirectional_iterator
      : std::integral_constant<bool,
            detail::belongs_to_iterator_category<
                typename std::decay<Iter>::type, std::bidirectional_iterator_tag
            >::value
         || detail::belongs_to_iterator_traversal<
                typename std::decay<Iter>::type, boost::bidirectional_traversal_tag
            >::value>
    {};

    template <typename Iter, typename Enable = void>
    struct is_random_access_iterator
      : std::integral_constant<bool,
            detail::has_category<
                typename std::decay<Iter>::type, std::random_access_iterator_tag
            >::value
         || detail::has_traversal<
                typename std::decay<Iter>::type, boost::random_access_traversal_tag
            >::value>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator, typename Enable = void>
    struct is_segmented_iterator;

    template <typename Iterator, typename Enable = void>
    struct is_segmented_local_iterator;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename Enable = void>
    struct is_zip_iterator
      : std::false_type
    {};
}}

#endif

