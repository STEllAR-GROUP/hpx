//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_LAUNCH_POLICY_APR_8_2014_0436PM)
#define HPX_TRAITS_IS_LAUNCH_POLICY_APR_8_2014_0436PM

#include <hpx/config.hpp>
#include <hpx/traits.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/util/decay.hpp>

#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/mpl/or.hpp>

namespace hpx { namespace threads
{
    class executor;
}}

namespace hpx { namespace traits
{
    namespace detail
    {
        template <typename Policy>
        struct is_launch_policy
          : boost::is_same<launch, Policy>
        {};

        template <typename Policy>
        struct is_threads_executor
          : boost::is_base_of<threads::executor, Policy>
        {};
    }

    template <typename Policy>
    struct is_launch_policy
      : detail::is_launch_policy<typename hpx::util::decay<Policy>::type>
    {};

    template <typename Policy>
    struct is_threads_executor
      : detail::is_threads_executor<typename hpx::util::decay<Policy>::type>
    {};

    template <typename Policy>
    struct is_launch_policy_or_executor
      : boost::mpl::or_<is_launch_policy<Policy>, is_threads_executor<Policy> >
    {};
}}

#endif

