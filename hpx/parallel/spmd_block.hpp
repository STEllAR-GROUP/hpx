//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SPMD_BLOCK_HPP)
#define HPX_PARALLEL_SPMD_BLOCK_HPP

#include <hpx/include/parallel_for_each.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/barrier.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/traits/is_execution_policy.hpp>

#include <boost/range/irange.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v2)
{
    namespace detail
    {
        template <typename T>
        struct extract_first_parameter
        {};

        // Specialization for lambdas
        template <typename ClassType, typename ReturnType>
        struct extract_first_parameter<ReturnType(ClassType::*)() const>
        {
            using type = std::false_type;
        };

        // Specialization for lambdas
        template <typename ClassType,
            typename ReturnType, typename Arg0, typename... Args>
        struct extract_first_parameter<
            ReturnType(ClassType::*)(Arg0, Args...) const>
        {
            using type = typename std::decay<Arg0>::type;
        };
    }

    /// The class spmd_block defines an interface for launching
    /// multiple images while giving handles to each image to interact with
    /// the remaining images. The \a define_spmd_block function templates create
    /// multiple images of a user-defined lambda and launches them in a possibly
    /// separate thread. A temporary spmd block object is created and diffused
    /// to each image. The constraint for the lambda given to the
    /// define_spmd_block function is to accept a spmd_block as first parameter.
    struct spmd_block
    {
        explicit spmd_block(std::size_t num_images, std::size_t image_id,
            hpx::lcos::local::barrier & barrier)
        : num_images_(num_images), image_id_(image_id), barrier_(barrier)
        {}

        // Note: spmd_block class is movable/move-assignable
        // but not copyable/copy-assignable

        spmd_block(spmd_block &&) = default;
        spmd_block(spmd_block const &) = delete;

        spmd_block & operator=(spmd_block &&) = default;
        spmd_block & operator=(spmd_block const &) = delete;

        std::size_t get_num_images() const
        {
            return num_images_;
        }

        std::size_t this_image() const
        {
            return image_id_;
        }

        void sync_all() const
        {
           barrier_.get().wait();
        }

    private:
        std::size_t num_images_;
        std::size_t image_id_;
        mutable std::reference_wrapper<hpx::lcos::local::barrier> barrier_;
    };

    namespace detail
    {
        template <typename F>
        struct spmd_block_helper
        {
            mutable std::shared_ptr<hpx::lcos::local::barrier> barrier_;
            typename std::decay<F>::type f_;
            std::size_t num_images_;

            template <typename ... Ts>
            void operator()(std::size_t image_id, Ts && ... ts) const
            {
                spmd_block block(num_images_, image_id, *barrier_);
                hpx::util::invoke(
                    f_, std::move(block), std::forward<Ts>(ts)...);
            }
        };
    }

    // Asynchronous version
    template <typename ExPolicy, typename F, typename ... Args,
        typename = std::enable_if_t<
            hpx::parallel::v1::is_async_execution_policy<ExPolicy>::value>
        >
    std::vector<hpx::future<void>>
    define_spmd_block(ExPolicy && policy,
        std::size_t num_images, F && f, Args && ... args)
    {
        static_assert(
            parallel::execution::is_execution_policy<ExPolicy>::value,
            "parallel::execution::is_execution_policy<ExPolicy>::value");

        using ftype = typename std::decay<F>::type;

        using first_type =
            typename hpx::parallel::v2::detail::extract_first_parameter<
                        decltype(&ftype::operator())>::type;

        using executor_type =
            typename hpx::util::decay<ExPolicy>::type::executor_type;

        static_assert(std::is_same<spmd_block, first_type>::value,
            "define_spmd_block() needs a lambda that " \
            "has at least a spmd_block as 1st argument");

        std::shared_ptr<hpx::lcos::local::barrier> barrier
            = std::make_shared<hpx::lcos::local::barrier>(num_images);

        return
            hpx::parallel::executor_traits<
                typename std::decay<executor_type>::type
                >::bulk_async_execute(
                    policy.executor(),
                    detail::spmd_block_helper<F>{
                        barrier, std::forward<F>(f), num_images
                    },
                    boost::irange(0ul, num_images), std::forward<Args>(args)...);
    }

    // Synchronous version
    template <typename ExPolicy, typename F, typename ... Args,
        typename = std::enable_if_t<
            !hpx::parallel::v1::is_async_execution_policy<ExPolicy>::value>
        >
    void
    define_spmd_block(ExPolicy && policy,
        std::size_t num_images, F && f, Args && ... args)
    {
        static_assert(
            parallel::execution::is_execution_policy<ExPolicy>::value,
            "parallel::execution::is_execution_policy<ExPolicy>::value");

        using ftype = typename std::decay<F>::type;

        using first_type =
            typename hpx::parallel::v2::detail::extract_first_parameter<
                        decltype(&ftype::operator())>::type;

        using executor_type =
            typename hpx::util::decay<ExPolicy>::type::executor_type;

        static_assert(std::is_same<spmd_block, first_type>::value,
            "define_spmd_block() needs a lambda that " \
            "has at least a spmd_block as 1st argument");

        std::shared_ptr<hpx::lcos::local::barrier> barrier
            = std::make_shared<hpx::lcos::local::barrier>(num_images);

        hpx::parallel::executor_traits<
            typename std::decay<executor_type>::type
            >::bulk_execute(
                policy.executor(),
                detail::spmd_block_helper<F>{
                    barrier, std::forward<F>(f), num_images
                },
                boost::irange(0ul, num_images), std::forward<Args>(args)...);
    }

    template <typename F, typename ... Args>
    void define_spmd_block(std::size_t num_images, F && f, Args && ... args)
    {
        define_spmd_block(parallel::execution::par,
            num_images, std::forward<F>(f), std::forward<Args>(args)...);
    }
}}}

#endif
