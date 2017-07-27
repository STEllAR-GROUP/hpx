//  Copyright (c) 2017 Hartmut Kaiser
//  Copyright (c) 2017 Google
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nounnamed

#if !defined(HPX_PARALLEL_EXECUTORS_EXECUTION_INFORMATION_FWD_JAN_16_2017_0350PM)
#define HPX_PARALLEL_EXECUTORS_EXECUTION_INFORMATION_FWD_JAN_16_2017_0350PM

#include <hpx/config.hpp>
#include <hpx/parallel/executors/execution_fwd.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/traits/executor_traits.hpp>

#include <cstddef>

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    // Define infrastructure for customization points
    namespace detail
    {
        struct processing_units_count_tag {};
        struct has_pending_closures_tag {};
        struct get_pu_mask_tag {};
        struct set_scheduler_mode_tag {};

#ifdef HPX_HAVE_CXX11_AUTO_RETURN_VALUE
        // forward declare customization point implementations
        template <>
        struct customization_point<processing_units_count_tag>
        {
            template <typename Executor, typename Parameters>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, Parameters& params) const;
        };

        template <>
        struct customization_point<has_pending_closures_tag>
        {
            template <typename Executor>
            HPX_FORCEINLINE
            auto operator()(Executor && exec) const;
        };

        template <>
        struct customization_point<get_pu_mask_tag>
        {
            template <typename Executor>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, threads::topology& topo,
                std::size_t thread_num) const;
        };

        template <>
        struct customization_point<set_scheduler_mode_tag>
        {
            template <typename Executor, typename Mode>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, Mode const& mode) const;
        };
#endif
    }
    /// \endcond
}}}

#endif

