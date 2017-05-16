//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SPMD_BLOCK_HPP)
#define HPX_PARALLEL_SPMD_BLOCK_HPP

#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/spmd_block.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/traits/is_execution_policy.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v2
{
    /// The class spmd_block defines an interface for launching
    /// multiple images while giving handles to each image to interact with
    /// the remaining images. The \a define_spmd_block function templates create
    /// multiple images of a user-defined lambda and launches them in a possibly
    /// separate thread. A temporary spmd block object is created and diffused
    /// to each image. The constraint for the lambda given to the
    /// define_spmd_block function is to accept a spmd_block as first parameter.
    using spmd_block = hpx::lcos::local::spmd_block;

    // Asynchronous version
    template <typename ExPolicy, typename F, typename ... Args,
        typename = typename std::enable_if<
            hpx::parallel::v1::is_async_execution_policy<ExPolicy>::value>::type
        >
    inline std::vector<hpx::future<void>>
    define_spmd_block(ExPolicy && policy,
        std::size_t num_images, F && f, Args && ... args)
    {
        return hpx::lcos::local::define_spmd_block(
            std::forward<ExPolicy>(policy), num_images,
                std::forward<F>(f), std::forward<Args>(args)...);
    }

    // Synchronous version
    template <typename ExPolicy, typename F, typename ... Args,
        typename = typename std::enable_if<
            !hpx::parallel::v1::is_async_execution_policy<ExPolicy>::value>::type
        >
    inline void
    define_spmd_block(ExPolicy && policy,
        std::size_t num_images, F && f, Args && ... args)
    {
        hpx::lcos::local::define_spmd_block(
            std::forward<ExPolicy>(policy), num_images,
                std::forward<F>(f), std::forward<Args>(args)...);
    }

    template <typename F, typename ... Args>
    inline void define_spmd_block(std::size_t num_images,
        F && f, Args && ... args)
    {
        hpx::lcos::local::define_spmd_block(parallel::execution::par,
            num_images, std::forward<F>(f), std::forward<Args>(args)...);
    }
}}}

#endif
