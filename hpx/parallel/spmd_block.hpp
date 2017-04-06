////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_PARALLEL_SPMD_BLOCK_HPP)
#define HPX_PARALLEL_SPMD_BLOCK_HPP

#include <hpx/include/parallel_for_each.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming/name.hpp>
/*#include <hpx/runtime/serialization/serialize.hpp>*/
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/barrier.hpp>

/*#include <hpx/runtime/actions/lambda_to_action.hpp>*/

#include <boost/range/irange.hpp>

#include <memory>

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
            using type = Arg0;
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

        spmd_block(
            std::string name, std::size_t num_images, std::size_t image_id)
        : name_(name), num_images_(num_images), image_id_(image_id)
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
           if (!barrier_)
           {
               barrier_ = std::make_shared<hpx::lcos::barrier>(
                 name_ + "_barrier"
               , num_images_
               );
           }

           barrier_->wait();
        }

        hpx::future<void> sync_all(hpx::launch::async_policy const &) const
        {
           if (!barrier_)
           {
               barrier_ = std::make_shared<hpx::lcos::barrier>(
                 name_+ "_barrier"
               , num_images_
               );
           }

           return barrier_->wait(hpx::launch::async);
        }

    private:
        std::string name_;
        std::size_t num_images_;
        std::size_t image_id_;
        mutable std::shared_ptr<hpx::lcos::barrier> barrier_;

/*    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const)
        {
            ar & name_ & num_images_ && image_id_;
        }*/
    };

    template <typename ExPolicy, typename F, typename ... Args>
    void define_spmd_block(
        ExPolicy && policy,
        std::string && name,
        std::size_t num_images,
        F && f, Args && ... args)
    {
        using ftype = typename std::remove_reference<F>::type;

        using first_type
            = typename
                hpx::parallel::v2::detail::extract_first_parameter<
                    decltype(&ftype::operator())>::type;

        static_assert( std::is_same<spmd_block,first_type>::value,
            "define_spmd_block() needs a lambda that " \
            "has at least a spmd_block as 1st argument");

        auto range = boost::irange(0ul, num_images);

// FIXME : How to invoke images remotely?
/*        auto a = hpx::actions::lambda_to_action(std::move(f));*/

        hpx::parallel::for_each(
            policy, range.begin(), range.end(),
            [=,&f](std::size_t image_id)
            {
                spmd_block block(name,num_images,image_id);
                f(block,args...);
            });
    }
}}}

#endif
