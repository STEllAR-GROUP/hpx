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


#include <boost/range/irange.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

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
        explicit spmd_block(std::size_t num_images, std::size_t image_id)
        : num_images_(num_images), image_id_(image_id), barrier_(num_images_+1)
        {}

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

    // FIXME : If 2 images are scheduled in the same thread (not OS-thread)
    // -> deadlock
        void sync_all() const
        {
           barrier_.wait();
        }

    private:
        std::size_t num_images_;
        std::size_t image_id_;
        mutable hpx::lcos::local::barrier barrier_;
    };

    namespace detail
    {
        template <typename F>
        struct spmd_block_helper
        {
            typename std::decay<F>::type f_;
            std::size_t num_images_;

            template <typename ... Ts>
            void operator()(std::size_t image_id, Ts && ... ts) const
            {

                spmd_block block(num_images_, image_id);
                hpx::util::invoke_r<void>(
                    f_, block, std::forward<Ts>(ts)...);
            }
        };
    }

    template <typename Executor, typename F, typename ... Args>
    void define_spmd_block(Executor && exec, std::size_t num_images,
        F && f, Args && ... args)
    {
        using ftype = typename std::decay<F>::type;

        using first_type =
                typename hpx::parallel::v2::detail::extract_first_parameter<
                            decltype(&ftype::operator())>::type;

        static_assert(std::is_same<spmd_block, first_type>::value,
            "define_spmd_block() needs a lambda that " \
            "has at least a spmd_block as 1st argument");

        hpx::parallel::executor_traits<
                typename std::decay<Executor>::type
            >::bulk_execute(
                std::forward<Executor>(exec),
                detail::spmd_block_helper<F>{
                    std::forward<F>(f), num_images
                },
                boost::irange(0ul, num_images), std::forward<Args>(args)...);
    }

    template <typename F, typename ... Args>
    void define_spmd_block(std::size_t num_images, F && f, Args && ... args)
    {
        define_spmd_block(hpx::parallel::parallel_executor(),
            num_images, std::forward<F>(f),
            std::forward<Args>(args)...);
    }
}}}

#endif
