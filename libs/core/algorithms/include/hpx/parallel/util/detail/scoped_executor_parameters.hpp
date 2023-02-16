//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>

namespace hpx::parallel::util::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Parameters, typename Executor>
    struct scoped_executor_parameters
    {
    public:
        scoped_executor_parameters(
            Parameters const& params, Executor const& exec)
          : params_(params)
          , exec_(exec)
        {
            execution::mark_begin_execution(params_, exec_);
        }

        void mark_end_of_scheduling() const
        {
            execution::mark_end_of_scheduling(params_, exec_);
        }

        ~scoped_executor_parameters()
        {
            execution::mark_end_execution(params_, exec_);
        }

    private:
        Parameters params_;
        Executor exec_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Parameters, typename Executor>
    struct scoped_executor_parameters_ref
    {
    public:
        scoped_executor_parameters_ref(
            Parameters const& params, Executor const& exec)
          : params_(params)
          , exec_(exec)
        {
            execution::mark_begin_execution(params_, exec_);
        }

        void mark_end_of_scheduling() const
        {
            execution::mark_end_of_scheduling(params_, exec_);
        }

        ~scoped_executor_parameters_ref()
        {
            execution::mark_end_execution(params_, exec_);
        }

    private:
        Parameters const& params_;
        Executor const& exec_;
    };
}    // namespace hpx::parallel::util::detail
