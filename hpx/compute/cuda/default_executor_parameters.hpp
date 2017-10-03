//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_CUDA_DEFAULT_EXECUTOR_PARAMETERS_HPP
#define HPX_COMPUTE_CUDA_DEFAULT_EXECUTOR_PARAMETERS_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)
#include <hpx/traits/is_executor_parameters.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace compute { namespace cuda
{
    struct default_executor_parameters
    {
        template <typename Executor, typename F>
        std::size_t get_chunk_size(Executor& exec, F &&, std::size_t cores,
            std::size_t num_tasks)
        {
            return std::size_t(-1);
        }
    };
}}}

namespace hpx { namespace parallel { namespace execution
{
    template <>
    struct is_executor_parameters<compute::cuda::default_executor_parameters>
      : std::true_type
    {};
}}}

#endif
#endif
