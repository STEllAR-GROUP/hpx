//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_IS_RANGE_JUL_18_2015_1017AM)
#define HPX_PARALLEL_TRAITS_IS_RANGE_JUL_18_2015_1017AM

#include <hpx/config.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/always_void.hpp>

#include <boost/range/iterator_range_core.hpp>

#include <type_traits>

namespace hpx { namespace parallel { namespace traits
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Enable = void>
        struct has_value_type
          : std::false_type
        {};

        template <typename T>
        struct has_value_type<T,
                typename hpx::util::always_void<typename T::value_type>::type>
          : std::true_type
        {};

        template <typename T, typename Enable = void>
        struct has_iterator
          : std::false_type
        {};

        template <typename T>
        struct has_iterator<T,
                typename hpx::util::always_void<typename T::iterator>::type>
          : std::true_type
        {};

        template <typename T, typename Enable = void>
        struct has_size_type
          : std::false_type
        {};

        template <typename T>
        struct has_size_type<T,
                typename hpx::util::always_void<typename T::size_type>::type>
          : std::true_type
        {};

        template <typename T, typename Enable = void>
        struct has_reference
          : std::false_type
        {};

        template <typename T>
        struct has_reference<T,
                typename hpx::util::always_void<typename T::reference>::type>
          : std::true_type
        {};

        template <typename T, typename Enable = void>
        struct is_container
          : std::integral_constant<bool,
                has_value_type<T>::value && has_iterator<T>::value &&
                has_size_type<T>::value && has_reference<T>::value>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Enable = void>
        struct is_iterator_range
          : std::false_type
        {};

        template <typename Iterator>
        struct is_iterator_range<boost::iterator_range<Iterator> >
          : std::true_type
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Rng, typename Enable = void>
        struct is_range
          : is_container<Rng>
        {};

        template <typename Iterator>
        struct is_range<boost::iterator_range<Iterator> >
          : std::true_type
        {};
    }

    template <typename Rng, typename Enable = void>
    struct is_range
      : detail::is_range<typename hpx::util::decay<Rng>::type>
    {};
}}}

#endif
