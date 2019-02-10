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

        template <typename T, typename U, typename = void>
        struct addition_result
        {};

        template <typename T, typename U>
        struct addition_result<T, U,
            typename util::always_void<
                decltype(std::declval<T>() + std::declval<U>())
            >::type>
        {
          using type = decltype(std::declval<T>() + std::declval<U>());
        };

        template <typename T, typename = void>
        struct dereference_result
        {};

        template <typename T>
        struct dereference_result<T,
            typename util::always_void< decltype(*(std::declval<T&>()))>::type>
        {
          using type = decltype(*(std::declval<T&>()));
        };

        template <typename T, typename U, typename = void>
        struct inplace_addition_result
        {};

        template <typename T, typename U>
        struct inplace_addition_result<T, U,
            typename util::always_void<
                decltype(std::declval<T>() += std::declval<U>())
            >::type>
        {
          using type = decltype(std::declval<T>() += std::declval<U>());
        };

        template <typename T, typename U, typename = void>
        struct inplace_subtraction_result
        {};

        template <typename T, typename U>
        struct inplace_subtraction_result<T, U,
            typename util::always_void<
                decltype(std::declval<T>() -= std::declval<U>())
            >::type>
        {
          using type = decltype(std::declval<T>() -= std::declval<U>());
        };

        template <typename T, typename = void>
        struct predecrement_result
        {};

        template <typename T>
        struct predecrement_result<T,
            typename util::always_void<decltype(--std::declval<T&>())>::type>
        {
          using type = decltype(--std::declval<T&>());
        };

        template <typename T, typename = void>
        struct postdecrement_result
        {};

        template <typename T>
        struct postdecrement_result<T,
            typename util::always_void<decltype(std::declval<T&>()--)>::type>
        {
          using type = decltype(std::declval<T&>()--);
        };

        template <typename T, typename U, typename = void>
        struct subscript_result
        {};

        template <typename T, typename U>
        struct subscript_result<T, U,
            typename util::always_void<
                decltype(std::declval<T&>()[std::declval<U>()])
            >::type>
        {
          using type = decltype(std::declval<T&>()[std::declval<U>()]);
        };

        template <typename T, typename U, typename = void>
        struct subtraction_result
        {};

        template <typename T, typename U>
        struct subtraction_result<T, U,
            typename util::always_void<
                decltype(std::declval<T>() - std::declval<U>())
            >::type>
        {
          using type = decltype(std::declval<T>() - std::declval<U>());
        };

        template <typename Iter, typename TraversalTag>
        struct satisfy_traversal_concept
          : std::false_type
        {};

        template <typename Iter>
        struct satisfy_traversal_concept<
            Iter, boost::bidirectional_traversal_tag>
          : std::integral_constant<bool,
                std::is_same<
                    typename std::add_lvalue_reference<Iter>::type,
                    typename detail::predecrement_result<Iter>::type
                >::value
                && std::is_same<Iter,
                    typename detail::postdecrement_result<Iter>::type
                >::value>
        {};

        template <typename Iter>
        struct satisfy_traversal_concept<Iter, boost::random_access_traversal_tag>
          : std::integral_constant<bool,
                satisfy_traversal_concept<Iter,
                    boost::bidirectional_traversal_tag
                >::value
                && std::is_same<
                    typename detail::dereference_result<Iter>::type,
                    typename detail::subscript_result<Iter,
                        typename std::iterator_traits<Iter>::difference_type
                    >::type
                >::value
                && std::is_same<
                    Iter,
                    typename detail::addition_result<Iter,
                        typename std::iterator_traits<Iter>::difference_type
                    >::type
                >::value
                && std::is_same<
                    typename std::add_lvalue_reference<Iter>::type,
                    typename detail::inplace_addition_result<Iter,
                        typename std::iterator_traits<Iter>::difference_type
                    >::type
                >::value
                && std::is_same<
                    Iter,
                    typename detail::subtraction_result<Iter,
                        typename std::iterator_traits<Iter>::difference_type
                    >::type
                >::value
                && std::is_same<
                    typename std::iterator_traits<Iter>::difference_type,
                    typename detail::subtraction_result<Iter, Iter>::type
                >::value
                && std::is_same<
                    typename std::add_lvalue_reference<Iter>::type,
                    typename detail::inplace_subtraction_result<Iter,
                        typename std::iterator_traits<Iter>::difference_type
                    >::type
                >::value>
        {};
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
          : std::integral_constant<bool,
                std::is_base_of<
                    Traversal, typename boost::iterator_traversal<Iter>::type
                >::value
                || satisfy_traversal_concept<Iter, Traversal>::value>
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

        template <typename Iter>
        struct has_traversal<Iter, boost::bidirectional_traversal_tag,
                typename std::enable_if<is_iterator<Iter>::value>::type>
          : std::integral_constant<bool,
                std::is_same<
                    boost::bidirectional_traversal_tag,
                    typename boost::iterator_traversal<Iter>::type
                >::value
                || (satisfy_traversal_concept<Iter,
                        boost::bidirectional_traversal_tag
                    >::value
                    && !satisfy_traversal_concept<Iter,
                        boost::random_access_traversal_tag
                    >::value)>
        {};

        template <typename Iter>
        struct has_traversal<Iter, boost::random_access_traversal_tag,
                typename std::enable_if<is_iterator<Iter>::value>::type>
          : std::integral_constant<bool,
                std::is_same<
                    boost::random_access_traversal_tag,
                    typename boost::iterator_traversal<Iter>::type
                >::value
                || satisfy_traversal_concept<Iter,
                       boost::random_access_traversal_tag
                   >::value>
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

