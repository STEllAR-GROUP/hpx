//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_TIMED_EXECUTOR_AUG_05_2015_0840AM)
#define HPX_TRAITS_IS_TIMED_EXECUTOR_AUG_05_2015_0840AM

#include <hpx/traits.hpp>
#include <hpx/config/inline_namespace.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>

#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    struct timed_executor_tag : executor_tag {};

    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_timed_executor
          : boost::is_base_of<timed_executor_tag, T>
        {};

        template <>
        struct is_timed_executor<timed_executor_tag>
          : std::false_type
        {};
        /// \endcond
    }

    template <typename T>
    struct is_timed_executor
      : detail::is_timed_executor<typename hpx::util::decay<T>::type>
    {};

    template <typename Executor, typename Enable = void>
    struct timed_executor_traits;
}}}

namespace hpx { namespace traits
{
    // new executor framework
    template <typename Executor, typename Enable>
    struct is_timed_executor
      : parallel::v3::is_timed_executor<Executor>
    {};
}}

#endif

