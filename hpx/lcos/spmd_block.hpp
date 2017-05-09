//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_SPMD_BLOCK_HPP)
#define HPX_LCOS_SPMD_BLOCK_HPP

#include <hpx/include/plain_actions.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/barrier.hpp>
#include <hpx/lcos/broadcast.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/runtime/actions/lambda_to_action.hpp>
#include <hpx/runtime/get_locality_id.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_execution_policy.hpp>
#include <hpx/util/unused.hpp>

#include <boost/range/irange.hpp>

#include <cstddef>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace lcos
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

        // Specialization for actions
        template <>
        struct extract_first_parameter< hpx::util::tuple<> >
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

        // Specialization for actions
        template <typename Arg0, typename... Args>
        struct extract_first_parameter< hpx::util::tuple<Arg0, Args...> >
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
        spmd_block(){}

        explicit spmd_block(std::string name, std::size_t num_images)
        : name_(name), num_images_(num_images), image_id_(hpx::get_locality_id())
        {}

        explicit spmd_block(std::string name, std::size_t num_images,
            std::size_t image_id)
        : name_(name), num_images_(num_images), image_id_(image_id)
        , barrier_(
            std::make_shared<hpx::lcos::barrier>(
                name_ + "_barrier", num_images_, image_id_))
        {}

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
            if(!barrier_)
            {
                barrier_ =
                    std::make_shared<hpx::lcos::barrier>(
                    name_ + "_barrier", num_images_, image_id_);
            }

           barrier_->wait();
        }

        hpx::future<void> sync_all(hpx::launch::async_policy const &) const
        {
            if(!barrier_)
            {
                barrier_ =
                    std::make_shared<hpx::lcos::barrier>(
                    name_ + "_barrier", num_images_, image_id_);
            }

           return barrier_->wait(hpx::launch::async);
        }

    private:
        std::string name_;
        std::size_t num_images_;
        std::size_t image_id_;
        mutable std::shared_ptr<hpx::lcos::barrier> barrier_;

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void save(Archive& ar, unsigned const) const
        {
            ar << name_ << num_images_;
        }

        template <typename Archive>
        void load(Archive& ar, const unsigned int)
        {
            ar >> name_ >> num_images_;
            image_id_ = hpx::get_locality_id();
        }

        HPX_SERIALIZATION_SPLIT_MEMBER()
    };

    // Helpers for bulk_execute() invoked in define_spmd_block()
    namespace detail
    {
        template <typename F, bool Condition =
            hpx::traits::is_action<F>::value >
        struct spmd_block_helper;

        // Overload for actions
        template <typename F>
        struct spmd_block_helper<F,true>
        {
            std::string name_;
            std::size_t num_images_;

            template <typename ... Ts>
            void operator()(std::size_t image_id, Ts && ... ts) const
            {
                spmd_block block(name_, num_images_, image_id);

                hpx::async<F>(
                    hpx::find_here(),
                    std::move(block),
                    std::forward<Ts>(ts)...).get();
            }
        };

        // Overload for lambdas
        template <typename F>
        struct spmd_block_helper<F,false>
        {
            std::string name_;
            std::size_t num_images_;

            template <typename ... Ts>
            void operator()(std::size_t image_id, Ts && ... ts) const
            {
                spmd_block block(name_, num_images_, image_id);

                int * dummy = nullptr;
                hpx::util::invoke(
                    reinterpret_cast<const F&>(*dummy),
                    std::move(block),
                    std::forward<Ts>(ts)...);
            }
        };
    }

    // Overload for lambdas (Note : it implies some undefined behaviour)
    template <typename F, typename ... Args>
    typename std::enable_if_t<!hpx::traits::is_action<F>::value,
        hpx::future<void> >
    define_spmd_block(std::string && name, std::size_t images_per_locality,
        F && f, Args && ... args)
    {
        using ftype = typename std::decay<F>::type;

        using first_type =
            typename hpx::lcos::detail::extract_first_parameter<
                        decltype(&ftype::operator())>::type;

        using executor_type =
            hpx::parallel::execution::parallel_executor;

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

        return
            hpx::lcos::broadcast(
                act, hpx::find_all_localities(),
                    std::forward<std::string>(name), images_per_locality,
                        num_images, std::forward<Args>(args)...);
    }

    // Helper for the action version of define_spmd_block()
    namespace detail
    {
        // Overload for actions
        template <typename F, typename ReturnType, typename ... Args>
        struct spmd_block_helper_action
        {
            using executor_type =
                hpx::parallel::execution::parallel_executor;

            static
            ReturnType call(
                std::string name,
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
                        detail::spmd_block_helper<F>{name,num_images},
                        boost::irange(
                            offset, offset + images_per_locality),
                        args...);
            }
        };
    }

    // Overload for actions
    template <typename F, typename ... Args>
    typename std::enable_if_t<hpx::traits::is_action<F>::value,
        hpx::future<void> >
    define_spmd_block(std::string && name, std::size_t images_per_locality,
        F && f, Args && ... args)
    {
        using action_type = typename std::decay<F>::type;

        using first_type =
            typename hpx::lcos::detail::extract_first_parameter<
                        typename action_type::arguments_type>::type;

        static_assert(std::is_same<spmd_block, first_type>::value,
            "define_spmd_block() needs an action that " \
            "has at least a spmd_block as 1st argument");

        using result_type = typename action_type::result_type;

        using helper_type =
            hpx::lcos::detail::spmd_block_helper_action<
                action_type, result_type,
                    typename std::decay<Args>::type... >;

        using helper_action_type =
            typename hpx::actions::make_action<
                decltype( &helper_type::call ), &helper_type::call >::type;

        helper_action_type act;

        std::size_t num_images
            = hpx::get_num_localities(hpx::launch::sync) * images_per_locality;

        return
            hpx::lcos::broadcast(
                act, hpx::find_all_localities(),
                    std::forward<std::string>(name), images_per_locality,
                        num_images, std::forward<Args>(args)...);
    }
}}

#endif
