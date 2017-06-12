//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_EXECUTOR_TRAITS_JAN_04_2017_0626PM)
#define HPX_TRAITS_EXECUTOR_TRAITS_JAN_04_2017_0626PM

#include <hpx/config.hpp>
#include <hpx/traits/has_member_xxx.hpp>
#include <hpx/util/detected.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v3
{
    struct static_chunk_size;
}}}

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    struct sequenced_execution_tag;
    struct parallel_execution_tag;
    struct unsequenced_execution_tag;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(post);
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(sync_execute);
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(async_execute);
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(then_execute);
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(bulk_sync_execute);
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(bulk_async_execute);
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(bulk_then_execute);
    }

    template <typename T, typename Enable = void>
    struct has_post_member
      : detail::has_post<typename std::decay<T>::type>
    {};

    template <typename T, typename Enable = void>
    struct has_sync_execute_member
      : detail::has_sync_execute<typename std::decay<T>::type>
    {};

    template <typename T, typename Enable = void>
    struct has_async_execute_member
      : detail::has_async_execute<typename std::decay<T>::type>
    {};

    template <typename T, typename Enable = void>
    struct has_then_execute_member
      : detail::has_then_execute<typename std::decay<T>::type>
    {};

    template <typename T, typename Enable = void>
    struct has_bulk_sync_execute_member
      : detail::has_bulk_sync_execute<typename std::decay<T>::type>
    {};

    template <typename T, typename Enable = void>
    struct has_bulk_async_execute_member
      : detail::has_bulk_async_execute<typename std::decay<T>::type>
    {};

    template <typename T, typename Enable = void>
    struct has_bulk_then_execute_member
      : detail::has_bulk_then_execute<typename std::decay<T>::type>
    {};

#if defined(HPX_HAVE_CXX17_VARIABLE_TEMPLATES)
    template <typename T>
    constexpr bool has_post_member_v =
        has_post_member<T>::value;

    template <typename T>
    constexpr bool has_sync_execute_member_v =
        has_sync_execute_member<T>::value;

    template <typename T>
    constexpr bool has_async_execute_member_v =
        has_async_execute_member<T>::value;

    template <typename T>
    constexpr bool has_then_execute_member_v =
        has_then_execute_member<T>::value;

    template <typename T>
    constexpr bool has_bulk_sync_execute_member_v =
        has_bulk_sync_execute_member<T>::value;

    template <typename T>
    constexpr bool has_bulk_async_execute_member_v =
        has_bulk_async_execute_member<T>::value;

    template <typename T>
    constexpr bool has_bulk_then_execute_member_v =
        has_bulk_then_execute_member<T>::value;
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_context
    {
        using type =
            typename std::decay<
                decltype(std::declval<Executor const&>().context())
            >::type;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Components which create groups of execution agents may use execution
    // categories to communicate the forward progress and ordering guarantees
    // of these execution agents with respect to other agents within the same
    // group.

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_execution_category
    {
    private:
        template <typename T>
        using execution_category = typename T::execution_category;

    public:
        using type = hpx::util::detected_or_t<
            unsequenced_execution_tag, execution_category, Executor>;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_shape
    {
    private:
        template <typename T>
        using shape_type = typename T::shape_type;

    public:
        using type = hpx::util::detected_or_t<
            std::size_t, shape_type, Executor>;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_index
    {
    private:
        // exposition only
        template <typename T>
        using index_type = typename T::index_type;

    public:
        using type = hpx::util::detected_or_t<
            typename executor_shape<Executor>::type, index_type, Executor>;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_parameters_type
    {
    private:
        template <typename T>
        using parameters_type = typename T::parameters_type;

    public:
        using type = hpx::util::detected_or_t<
            parallel::static_chunk_size, parameters_type, Executor>;
    };
}}}

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct has_post_member
      : parallel::execution::has_post_member<T>
    {};

    template <typename T, typename Enable = void>
    struct has_sync_execute_member
      : parallel::execution::has_sync_execute_member<T>
    {};

    template <typename T, typename Enable = void>
    struct has_async_execute_member
      : parallel::execution::has_async_execute_member<T>
    {};

    template <typename T, typename Enable = void>
    struct has_then_execute_member
      : parallel::execution::has_then_execute_member<T>
    {};

    template <typename T, typename Enable = void>
    struct has_bulk_sync_execute_member
      : parallel::execution::has_bulk_sync_execute_member<T>
    {};

    template <typename T, typename Enable = void>
    struct has_bulk_async_execute_member
      : parallel::execution::has_bulk_async_execute_member<T>
    {};

    template <typename T, typename Enable = void>
    struct has_bulk_then_execute_member
      : parallel::execution::has_bulk_then_execute_member<T>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor, typename Enable = void>
    struct executor_context
      : parallel::execution::executor_context<Executor>
    {};

    template <typename Executor>
    using executor_context_t = typename executor_context<Executor>::type;

    template <typename Executor, typename Enable = void>
    struct executor_execution_category
      : parallel::execution::executor_execution_category<Executor>
    {};

    template <typename Executor>
    using executor_execution_category_t =
        typename executor_execution_category<Executor>::type;

    template <typename Executor, typename Enable = void>
    struct executor_shape
      : parallel::execution::executor_shape<Executor>
    {};

    template <typename Executor>
    using executor_shape_t = typename executor_shape<Executor>::type;

    template <typename Executor, typename Enable = void>
    struct executor_index
      : parallel::execution::executor_index<Executor>
    {};

    template <typename Executor>
    using executor_index_t = typename executor_index<Executor>::type;

    ///////////////////////////////////////////////////////////////////////////
    // extension
    template <typename Executor, typename Enable = void>
    struct executor_parameters_type
      : parallel::execution::executor_parameters_type<Executor>
    {};

    template <typename Executor>
    using executor_parameters_type_t =
        typename executor_parameters_type<Executor>::type;
}}

#endif
