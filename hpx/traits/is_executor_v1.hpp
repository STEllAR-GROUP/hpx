//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_EXECUTOR_MAY_13_2015_1213PM)
#define HPX_TRAITS_IS_EXECUTOR_MAY_13_2015_1213PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
#include <hpx/config/inline_namespace.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    struct executor_tag {};

    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_executor
          : std::is_base_of<executor_tag, T>
        {};

        template <>
        struct is_executor<executor_tag>
          : std::false_type
        {};
        /// \endcond
    }

    template <typename T>
    struct is_executor
      : detail::is_executor<typename hpx::util::decay<T>::type>
    {};

    template <typename Executor, typename Enable = void>
    struct executor_traits;

    template <typename Executor, typename Enable = void>
    struct executor_information_traits;
}}}

namespace hpx { namespace traits
{
    // new executor framework
    template <typename Executor, typename Enable = void>
    struct is_executor
      : parallel::v3::is_executor<Executor>
    {};
}}

#endif
#endif

