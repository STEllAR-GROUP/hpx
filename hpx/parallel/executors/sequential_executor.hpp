//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/sequential_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_SEQUENTIAL_EXECUTOR_MAY_11_2015_1050AM)
#define HPX_PARALLEL_EXECUTORS_SEQUENTIAL_EXECUTOR_MAY_11_2015_1050AM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    struct sequential_executor
    {
        typedef sequential_execution_tag execution_category;

        template <typename F>
        typename util::result_of<F()>::type
        execute(F f)
        {
            return f();
        }

        template <typename F>
        hpx::future<typename util::result_of<F()>::type>
        async_execute(F f)
        {
            return hpx::async(hpx::launch::deferred, f);
        }

        template <typename F, typename Shape>
        void bulk_execute(F f, Shape const& shape)
        {
            for (auto const& elem: shape)
                f(elem);
        }

        template <typename F, typename Shape>
        hpx::future<void>
        bulk_async_execute(F f, Shape const& shape)
        {
            return hpx::async(hpx::launch::deferred,
                [=] { this->bulk_execute(f, shape); });
        }
    };

    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_executor<sequential_executor>
          : std::true_type
        {};
        // \endcond
    }
}}}

#endif
