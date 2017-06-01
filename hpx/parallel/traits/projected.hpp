//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_PROJECTED_JUL_18_2015_1001PM)
#define HPX_PARALLEL_TRAITS_PROJECTED_JUL_18_2015_1001PM

#include <hpx/config.hpp>
#include <hpx/parallel/traits/vector_pack_load_store.hpp>
#include <hpx/parallel/traits/vector_pack_type.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/is_execution_policy.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/result_of.hpp>

#include <iterator>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct projected_iterator
    {
        typedef typename hpx::util::decay<T>::type type;
    };

    // For segmented iterators, we consider the local_raw_iterator instead of
    // the given one.
    template <typename Iterator>
    struct projected_iterator<Iterator,
        typename std::enable_if<
            is_segmented_iterator<Iterator>::value
        >::type>
    {
        typedef typename segmented_iterator_traits<
                Iterator
            >::local_iterator local_iterator;

        typedef typename segmented_local_iterator_traits<
                local_iterator
            >::local_raw_iterator type;
    };

    template <typename Iterator>
    struct projected_iterator<Iterator,
        typename hpx::util::always_void<
            typename hpx::util::decay<Iterator>::type::proxy_type
        >::type>
    {
        typedef typename hpx::util::decay<Iterator>::type::proxy_type
            type;
    };
}}

namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename F, typename Iter, typename Enable = void>
        struct projected_result_of;

        template <typename Proj, typename Iter>
        struct projected_result_of<Proj, Iter,
                typename std::enable_if<
                    hpx::traits::is_iterator<Iter>::value
                >::type>
          : hpx::util::invoke_result<Proj,
                    typename std::iterator_traits<Iter>::reference
                >
        {};

        template <typename Projected>
        struct projected_result_of_indirect
          : projected_result_of<
                typename Projected::projector_type,
                typename Projected::iterator_type>
        {};

#if defined(HPX_HAVE_DATAPAR)
        // This is being instantiated if a vector pack execution policy is used
        // with a zip_iterator. In this case the function object is invoked
        // with a tuple<datapar<T>...> instead of just a tuple<T...>
        template <typename Proj, typename ValueType, typename Enable = void>
        struct projected_result_of_vector_pack_
          : hpx::util::invoke_result<Proj,
                    typename hpx::parallel::traits::vector_pack_load<
                        typename hpx::parallel::traits::vector_pack_type<
                            ValueType
                        >::type,
                        ValueType
                    >::value_type&
                >
        {};

        template <typename Projected, typename Enable = void>
        struct projected_result_of_vector_pack;

        template <typename Projected>
        struct projected_result_of_vector_pack<Projected,
                typename hpx::util::always_void<
                    typename Projected::iterator_type
                >::type>
          : projected_result_of_vector_pack_<
                typename Projected::projector_type,
                typename std::iterator_traits<
                    typename Projected::iterator_type
                >::value_type>
        {};
#endif
    }

    template <typename F, typename Iter, typename Enable = void>
    struct projected_result_of
      : detail::projected_result_of<
            typename hpx::util::decay<F>::type,
            typename hpx::util::decay<Iter>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename F, typename Iter, typename Enable = void>
        struct is_projected
          : std::false_type
        {};

        template <typename Proj, typename Iter>
        struct is_projected<Proj, Iter,
                typename std::enable_if<
                    hpx::traits::is_iterator<Iter>::value
                >::type>
          : hpx::traits::is_invocable<
                Proj, typename std::iterator_traits<Iter>::reference
            >
        {};

        template <typename Projected, typename Enable = void>
        struct is_projected_indirect
          : std::false_type
        {};

        template <typename Projected>
        struct is_projected_indirect<Projected,
                typename hpx::util::always_void<
                    typename Projected::projector_type
                >::type>
          : detail::is_projected<
                typename Projected::projector_type,
                typename Projected::iterator_type>
        {};
    }

    template <typename F, typename Iter, typename Enable = void>
    struct is_projected
      : detail::is_projected<
            typename hpx::util::decay<F>::type,
            typename hpx::traits::projected_iterator<Iter>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Proj, typename Iter>
    struct projected
    {
        typedef typename hpx::util::decay<Proj>::type projector_type;
        typedef typename hpx::traits::projected_iterator<Iter>::type
            iterator_type;
    };

    template <typename Projected, typename Enable = void>
    struct is_projected_indirect
      : detail::is_projected_indirect<Projected>
    {};

    template <typename Projected, typename Enable = void>
    struct is_projected_zip_iterator
      : std::false_type
    {};

    template <typename Projected>
    struct is_projected_zip_iterator<Projected,
            typename hpx::util::always_void<
                typename Projected::iterator_type
            >::type>
      : hpx::traits::is_zip_iterator<typename Projected::iterator_type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename F, typename ...Args>
        struct is_indirect_callable_impl
          : hpx::traits::is_invocable<F, Args...>
        {};

        template <typename ExPolicy, typename F, typename ProjectedPack,
            typename Enable = void>
        struct is_indirect_callable
          : std::false_type
        {};

        template <typename ExPolicy, typename F, typename ...Projected>
        struct is_indirect_callable<
                ExPolicy, F, hpx::util::detail::pack<Projected...>,
                typename std::enable_if<
                    hpx::util::detail::all_of<
                        is_projected_indirect<Projected>...
                    >::value &&
                    (  !hpx::parallel::execution::is_vectorpack_execution_policy<
                            ExPolicy
                        >::value ||
                       !hpx::util::detail::all_of<
                            is_projected_zip_iterator<Projected>...
                        >::value
                    )
                >::type>
          : is_indirect_callable_impl<
                F, typename projected_result_of_indirect<Projected>::type...>
        {};

#if defined(HPX_HAVE_DATAPAR)
        // Vector pack execution policies used with zip-iterators require
        // special handling because zip_iterator<>::reference is not a real
        // reference type.
        template <typename ExPolicy, typename F, typename ...Projected>
        struct is_indirect_callable<
                ExPolicy, F, hpx::util::detail::pack<Projected...>,
                typename std::enable_if<
                    hpx::util::detail::all_of<
                        is_projected_indirect<Projected>...
                    >::value &&
                    hpx::parallel::execution::is_vectorpack_execution_policy<
                        ExPolicy
                    >::value &&
                    hpx::util::detail::all_of<
                        is_projected_zip_iterator<Projected>...
                    >::value
                >::type>
          : is_indirect_callable_impl<
                F, typename projected_result_of_vector_pack<Projected>::type...>
        {};
#endif
    }

    template <typename ExPolicy, typename F, typename ...Projected>
    struct is_indirect_callable
      : detail::is_indirect_callable<
            typename hpx::util::decay<ExPolicy>::type,
            typename hpx::util::decay<F>::type,
            hpx::util::detail::pack<
                typename hpx::util::decay<Projected>::type...
            > >
    {};
}}}

#endif

