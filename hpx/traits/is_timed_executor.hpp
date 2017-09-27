//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_TIMED_EXECUTOR_AUG_05_2015_0840AM)
#define HPX_TRAITS_IS_TIMED_EXECUTOR_AUG_05_2015_0840AM

#include <hpx/config.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    struct timed_executor_tag : executor_tag {};

    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_timed_executor
          : std::false_type
        {};
        /// \endcond
    }

    // Executor type traits:

    // Condition: T meets the syntactic requirements for OneWayExecutor
    // Precondition: T is a complete type
    template <typename T>
    struct is_timed_executor
      : detail::is_timed_executor<typename hpx::util::decay<T>::type>
    {};

    template <typename T>
    using is_timed_executor_t = typename is_timed_executor<T>::type;

#if defined(HPX_HAVE_CXX17_VARIABLE_TEMPLATES)
    template <typename T>
    constexpr bool is_timed_executor_v = is_timed_executor<T>::value;
#endif
}}}

namespace hpx { namespace traits
{
    // new executor framework
    template <typename Executor, typename Enable = void>
    struct is_timed_executor
      : parallel::execution::is_timed_executor<Executor>
    {};
}}

#endif

