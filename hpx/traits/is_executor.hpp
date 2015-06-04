//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_EXECUTOR_MAY_13_2015_1213PM)
#define HPX_TRAITS_IS_EXECUTOR_MAY_13_2015_1213PM

#include <hpx/traits.hpp>
#include <hpx/config/inline_namespace.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_executor
          : std::false_type
        {};
        /// \endcond
    }

    template <typename T>
    struct is_executor
      : detail::is_executor<typename hpx::util::decay<T>::type>
    {};

    template <typename Executor, typename Enable = void>
    class executor_traits;
}}}

namespace hpx { namespace traits
{
    // new executor framework
    template <typename Executor, typename Enable>
    struct is_executor
      : parallel::v3::is_executor<Executor>
    {};
}}

#endif

