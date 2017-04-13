//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SPMD_BLOCK_HPP)
#define HPX_PARALLEL_SPMD_BLOCK_HPP

#include <hpx/lcos/future.hpp>
#include <hpx/lcos/barrier.hpp>
#include <hpx/lcos/broadcast.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/runtime/actions/lambda_to_action.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/traits/is_execution_policy.hpp>
#include <hpx/util/unused.hpp>

#include <boost/range/irange.hpp>

#include <cstddef>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>

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
            hpx::lcos::barrier & barrier)
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

        hpx::future<void> sync_all(hpx::launch::async_policy const &) const
        {
           return barrier_.get().wait(hpx::launch::async);
        }

    private:
        std::size_t num_images_;
        std::size_t image_id_;
        mutable std::reference_wrapper<hpx::lcos::barrier> barrier_;
    };

    namespace detail
    {
        template <typename F>
        struct spmd_block_helper
        {
            std::string name_;
            std::size_t num_images_;

            template <typename ... Ts>
            void operator()(std::size_t image_id, Ts && ... ts) const
            {
                hpx::lcos::barrier
                    barrier(name_ + "_barrier" , num_images_, image_id);
                spmd_block block(num_images_, image_id, barrier);

                int * dummy = nullptr;
                hpx::util::invoke(
                    reinterpret_cast<const F&>(*dummy),
                    std::move(block),
                    std::forward<Ts>(ts)...);

                // Ensure that other images reaches that point
                barrier.wait();
            }
        };
    }

    template <typename ExPolicy, typename F, typename ... Args>
    void define_spmd_block(std::string && name, ExPolicy &&,
        std::size_t images_per_locality, F && f, Args && ... args)
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

        std::size_t num_images
            = hpx::get_num_localities(hpx::launch::sync) * images_per_locality;

        // Force f to be initialized at compile-time
        auto dummy = hpx::actions::lambda_to_action(f);
        HPX_UNUSED(dummy);

        auto act = hpx::actions::lambda_to_action(
            []( std::string name,
                std::size_t images_per_locality,
                std::size_t num_images,
                Args... args)
            {
                executor_type exec;
                std::size_t offset = hpx::get_locality_id();
                offset *= images_per_locality;

                hpx::parallel::executor_traits<
                    executor_type
                    >::bulk_execute(
                        exec,
                        detail::spmd_block_helper<ftype>{name,num_images},
                        boost::irange(
                            offset, offset + images_per_locality),
                        args...);
            });

        hpx::lcos::broadcast(
            act, hpx::find_all_localities(),
                std::forward<std::string>(name), images_per_locality, num_images,
                    std::forward<Args>(args)...);
    }

    template <typename F, typename ... Args>
    void define_spmd_block(std::string && name, std::size_t images_per_locality,
        F && f, Args && ... args)
    {
        define_spmd_block(std::forward<std::string>(name), parallel::execution::par,
            images_per_locality, std::forward<F>(f), std::forward<Args>(args)...);
    }
}}}

#endif
