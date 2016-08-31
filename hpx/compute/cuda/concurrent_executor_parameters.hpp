//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_CUDA_CONCURRENT_EXECUTOR_PARAMETERS_HPP
#define HPX_COMPUTE_CUDA_CONCURRENT_EXECUTOR_PARAMETERS_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)
#include <hpx/traits/is_executor_parameters.hpp>

#include <cstddef>

namespace hpx { namespace compute { namespace cuda
{
    struct concurrent_executor_parameters : parallel::executor_parameters_tag
    {
        template <typename Executor, typename F>
        std::size_t get_chunk_size(Executor& exec, F &&, std::size_t cores,
            std::size_t num_tasks)
        {
            return (num_tasks + cores - 1)/cores;
        }
    };
}}}

#endif
#endif
