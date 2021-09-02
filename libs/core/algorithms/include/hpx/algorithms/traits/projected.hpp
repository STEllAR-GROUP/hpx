//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/execution/traits/vector_pack_load_store.hpp>
#include <hpx/execution/traits/vector_pack_type.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/traits/is_invocable.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/type_support/always_void.hpp>
#include <hpx/type_support/pack.hpp>

#include <iterator>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct projected_iterator
    {
        using type = typename std::decay<T>::type;
    };

    // For segmented iterators, we consider the local_raw_iterator instead of
    // the given one.
    template <typename Iterator>
    struct projected_iterator<Iterator,
        typename std::enable_if<is_segmented_iterator<Iterator>::value>::type>
    {
        using local_iterator =
            typename segmented_iterator_traits<Iterator>::local_iterator;

        using type = typename segmented_local_iterator_traits<
            local_iterator>::local_raw_iterator;
    };

    template <typename Iterator>
    struct projected_iterator<Iterator,
        typename hpx::util::always_void<
            typename std::decay<Iterator>::type::proxy_type>::type>
    {
        using type = typename std::decay<Iterator>::type::proxy_type;
    };
}}    // namespace hpx::traits

namespace hpx { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename F, typename Iter, typename Enable = void>
        struct projected_result_of;

        template <typename Proj, typename Iter>
        struct projected_result_of<Proj, Iter,
            typename std::enable_if<
                hpx::traits::is_iterator<Iter>::value>::type>
          : hpx::util::invoke_result<Proj,
                typename std::iterator_traits<Iter>::reference>
        {
        };

        template <typename Projected>
        struct projected_result_of_indirect
          : projected_result_of<typename Projected::projector_type,
                typename Projected::iterator_type>
        {
        };

#if defined(HPX_HAVE_DATAPAR)
        // This is being instantiated if a vector pack execution policy is used
        // with a zip_iterator. In this case the function object is invoked
        // with a tuple<datapar<T>...> instead of just a tuple<T...>
        template <typename Proj, typename ValueType, typename Enable = void>
        struct projected_result_of_vector_pack_
          : hpx::util::invoke_result<Proj,
                typename hpx::parallel::traits::vector_pack_load<
                    typename hpx::parallel::traits::vector_pack_type<
                        ValueType>::type,
                    ValueType>::value_type&>
        {
        };

        template <typename Projected, typename Enable = void>
        struct projected_result_of_vector_pack;

        template <typename Projected>
        struct projected_result_of_vector_pack<Projected,
            typename hpx::util::always_void<
                typename Projected::iterator_type>::type>
          : projected_result_of_vector_pack_<typename Projected::projector_type,
                typename std::iterator_traits<
                    typename Projected::iterator_type>::value_type>
        {
        };
#endif
    }    // namespace detail

    template <typename F, typename Iter, typename Enable = void>
    struct projected_result_of
      : detail::projected_result_of<typename std::decay<F>::type,
            typename std::decay<Iter>::type>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename F, typename Iter, typename Enable = void>
        struct is_projected : std::false_type
        {
        };

        // the given projection function is valid, if it can be invoked using
        // the dereferenced iterator type and if the projection does not return
        // void

        // clang-format off
        template <typename Proj, typename Iter>
        struct is_projected<Proj, Iter,
            typename std::enable_if<
                hpx::traits::is_iterator<Iter>::value &&
                hpx::is_invocable<Proj,
                    typename std::iterator_traits<Iter>::reference>::value
             >::type>
          : std::integral_constant<bool,
                !std::is_void<typename hpx::util::invoke_result<Proj,
                    typename std::iterator_traits<Iter>::reference>::type
                >::value>
        {
        };
        // clang-format on

        template <typename Projected, typename Enable = void>
        struct is_projected_indirect : std::false_type
        {
        };

        template <typename Projected>
        struct is_projected_indirect<Projected,
            typename hpx::util::always_void<
                typename Projected::projector_type>::type>
          : detail::is_projected<typename Projected::projector_type,
                typename Projected::iterator_type>
        {
        };
    }    // namespace detail

    template <typename F, typename Iter, typename Enable = void>
    struct is_projected
      : detail::is_projected<typename std::decay<F>::type,
            typename hpx::traits::projected_iterator<Iter>::type>
    {
    };

    template <typename F, typename Iter>
    using is_projected_t = typename is_projected<F, Iter>::type;

    template <typename F, typename Iter>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_projected_v =
        is_projected<F, Iter>::value;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Proj, typename Iter>
    struct projected
    {
        using projector_type = typename std::decay<Proj>::type;
        using iterator_type =
            typename hpx::traits::projected_iterator<Iter>::type;
    };

    template <typename Projected, typename Enable = void>
    struct is_projected_indirect : detail::is_projected_indirect<Projected>
    {
    };

    template <typename Projected, typename Enable = void>
    struct is_projected_zip_iterator : std::false_type
    {
    };

    template <typename Projected>
    struct is_projected_zip_iterator<Projected,
        typename hpx::util::always_void<
            typename Projected::iterator_type>::type>
      : hpx::traits::is_zip_iterator<typename Projected::iterator_type>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename F, typename... Args>
        struct is_indirect_callable_impl : hpx::is_invocable<F, Args...>
        {
        };

        template <typename ExPolicy, typename F, typename ProjectedPack,
            typename Enable = void>
        struct is_indirect_callable : std::false_type
        {
        };

        template <typename ExPolicy, typename F, typename... Projected>
        struct is_indirect_callable<ExPolicy, F, hpx::util::pack<Projected...>,
            typename std::enable_if<
                hpx::util::all_of<is_projected_indirect<Projected>...>::value &&
                (!hpx::is_vectorpack_execution_policy<ExPolicy>::value ||
                    !hpx::util::all_of<
                        is_projected_zip_iterator<Projected>...>::value)>::type>
          : is_indirect_callable_impl<F,
                typename projected_result_of_indirect<Projected>::type...>
        {
        };

#if defined(HPX_HAVE_DATAPAR)
        // Vector pack execution policies used with zip-iterators require
        // special handling because zip_iterator<>::reference is not a real
        // reference type.
        template <typename ExPolicy, typename F, typename... Projected>
        struct is_indirect_callable<ExPolicy, F, hpx::util::pack<Projected...>,
            typename std::enable_if<
                hpx::util::all_of<is_projected_indirect<Projected>...>::value &&
                hpx::is_vectorpack_execution_policy<ExPolicy>::value &&
                hpx::util::all_of<
                    is_projected_zip_iterator<Projected>...>::value>::type>
          : is_indirect_callable_impl<F,
                typename projected_result_of_vector_pack<Projected>::type...>
        {
        };
#endif
    }    // namespace detail

    template <typename ExPolicy, typename F, typename... Projected>
    struct is_indirect_callable
      : detail::is_indirect_callable<typename std::decay<ExPolicy>::type,
            typename std::decay<F>::type,
            hpx::util::pack<typename std::decay<Projected>::type...>>
    {
    };

    template <typename ExPolicy, typename F, typename... Projected>
    using is_indirect_callable_t =
        typename is_indirect_callable<ExPolicy, F, Projected...>::type;

    template <typename ExPolicy, typename F, typename... Projected>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_indirect_callable_v =
        is_indirect_callable<ExPolicy, F, Projected...>::value;

}}}    // namespace hpx::parallel::traits
