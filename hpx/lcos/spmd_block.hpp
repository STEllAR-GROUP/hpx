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
#include <hpx/runtime/get_locality_id.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/first_argument.hpp>
#include <hpx/util/jenkins_hash.hpp>

#include <boost/range/irange.hpp>

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <set>
#include <type_traits>
#include <utility>

namespace hpx { namespace lcos
{
    /// The class spmd_block defines an interface for launching
    /// multiple images while giving handles to each image to interact with
    /// the remaining images. The \a define_spmd_block function templates create
    /// multiple images of a user-defined action and launches them in a possibly
    /// separate thread. A temporary spmd block object is created and diffused
    /// to each image. The constraint for the action given to the
    /// define_spmd_block function is to accept a spmd_block as first parameter.
    struct spmd_block
    {
    private:
        using barrier_type = hpx::lcos::barrier;
        using table_type =
            std::map<std::set<std::size_t>,std::shared_ptr<barrier_type>>;
    public:
        spmd_block(){}

        explicit spmd_block(std::string name, std::size_t images_per_locality,
            std::size_t num_images, std::size_t image_id)
        : name_(name), images_per_locality_(images_per_locality)
        , num_images_(num_images), image_id_(image_id)
        , barrier_(
            std::make_shared<hpx::lcos::barrier>(
                name_ + "_barrier", num_images_, image_id_))
        {}


        std::size_t get_images_per_locality() const
        {
            return images_per_locality_;
        }


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
           barrier_->wait();
        }

        hpx::future<void> sync_all(hpx::launch::async_policy const &) const
        {
           return barrier_->wait(hpx::launch::async);
        }

        // Synchronous versions of sync_images()

        void sync_images(std::set<std::size_t> const & images) const
        {
            using list_type = std::set<std::size_t>;

            typename table_type::iterator table_it(barriers_.find(images));
            typename list_type::iterator image_it(images.find(image_id_));

            // Is current image in the input list?
            if(image_it != images.end())
            {
                // Does the barrier for the input list non-exist?
                if(table_it == barriers_.end())
                {
                    std::size_t rank = std::distance(images.begin(),image_it);
                    std::string suffix;

                    for(std::size_t s : images)
                        suffix += ("_" + std::to_string(s));

                    table_it = barriers_.insert({images,
                        std::make_shared<barrier_type>(
                            name_ + "_barrier_" + std::to_string(hash_(suffix)),
                            images.size(),
                            rank)}).first;
                }

                table_it->second->wait();
            }
        }

        void sync_images(std::vector<std::size_t> const & input_images) const
        {
            std::set<std::size_t> images(
                input_images.begin(),input_images.end());
            sync_images(images);
        }

        template<typename Iterator>
        typename std::enable_if<
            traits::is_input_iterator<Iterator>::value
        >::type
        sync_images(Iterator begin, Iterator end) const
        {
            std::set<std::size_t> images(begin,end);
            sync_images(images);
        }

        template<typename ... I>
        typename std::enable_if<
            util::detail::all_of<
                typename std::is_integral<I>::type ... >::value
        >::type
        sync_images(I... i)
        {
            std::set<std::size_t> images = {(std::size_t)i...};
            sync_images(images);
        }

        // Asynchronous versions of sync_images()

        hpx::future<void>
        sync_images(hpx::launch::async_policy const & policy,
            std::set<std::size_t> const & images) const
        {
            using list_type = std::set<std::size_t>;

            typename table_type::iterator table_it(barriers_.find(images));
            typename list_type::iterator image_it(images.find(image_id_));

            // Is current image in the input list?
            if(image_it != images.end())
            {
                // Does the barrier for the input list non-exist?
                if(table_it == barriers_.end())
                {
                    std::size_t rank = std::distance(images.begin(),image_it);
                    std::string suffix;

                    for(std::size_t s : images)
                        suffix += ("_" + std::to_string(s));

                    table_it = barriers_.insert({images,
                        std::make_shared<barrier_type>(
                            name_ + "_barrier_" + std::to_string(hash_(suffix)),
                            images.size(),
                            rank)}).first;
                }

                return table_it->second->wait(hpx::launch::async);
            }

            return hpx::make_ready_future();
        }

        hpx::future<void>
        sync_images(hpx::launch::async_policy const & policy,
            std::vector<std::size_t> const & input_images) const
        {
            std::set<std::size_t> images(
                input_images.begin(),input_images.end());
            return sync_images(policy,images);
        }

        template<typename Iterator>
        typename std::enable_if<
            traits::is_input_iterator<Iterator>::value,
            hpx::future<void>
        >::type
        sync_images(hpx::launch::async_policy const & policy,
            Iterator begin, Iterator end) const
        {
            std::set<std::size_t> images(begin,end);
            return sync_images(policy,images);
        }

        template<typename ... I>
        typename std::enable_if<
            util::detail::all_of<
                typename std::is_integral<I>::type ... >::value,
            hpx::future<void>
        >::type
        sync_images(hpx::launch::async_policy const & policy,
            I ... i) const
        {
            std::set<std::size_t> images = {(std::size_t)i...};
            return sync_images(policy,images);
        }

    private:
        std::string name_;
        std::size_t images_per_locality_;
        std::size_t num_images_;
        std::size_t image_id_;
        hpx::util::jenkins_hash hash_;

        // Note : barrier is stored as a pointer because hpx::lcos::barrier
        // default constructor does not exist (Needed by
        // spmd_block::spmd_block())
        mutable std::shared_ptr<hpx::lcos::barrier> barrier_;
        mutable table_type barriers_;

    private:
        friend class hpx::serialization::access;

        // dummy serialization functionality
        template <typename Archive>
        void serialize(Archive &, unsigned) {}
    };

    // Helpers for bulk_execute() invoked in define_spmd_block()
    namespace detail
    {
        template <typename F>
        struct spmd_block_helper
        {
            std::string name_;
            std::size_t images_per_locality_;
            std::size_t num_images_;

            template <typename ... Ts>
            void operator()(std::size_t image_id, Ts && ... ts) const
            {
                using first_type =
                    typename hpx::util::first_argument<F>::type;

                static_assert(std::is_same<hpx::lcos::spmd_block,
                    first_type>::value,
                        "define_spmd_block() needs an action that " \
                        "has at least a spmd_block as 1st argument");

                hpx::lcos::spmd_block block(name_, images_per_locality_,
                    num_images_, image_id);

                F()(hpx::launch::sync,
                    hpx::find_here(),
                    std::move(block),
                    std::forward<Ts>(ts)...);
            }
        };
    }

    // Helper for define_spmd_block()
    namespace detail
    {
        template <typename F, typename ... Args>
        struct spmd_block_helper_action
        {
            static void call(
                std::string name,
                std::size_t images_per_locality,
                std::size_t num_images,
                Args... args)
            {
                using executor_type =
                    hpx::parallel::execution::parallel_executor;

                executor_type exec;
                std::size_t offset = hpx::get_locality_id();
                offset *= images_per_locality;

                hpx::parallel::executor_traits<
                    executor_type
                    >::bulk_execute(
                        exec,
                        detail::spmd_block_helper<F>{
                            name,images_per_locality, num_images},
                        boost::irange(
                            offset, offset + images_per_locality),
                        args...);
            }
        };
    }

    template <typename F, typename ... Args,
        HPX_CONCEPT_REQUIRES_(hpx::traits::is_action<F>::value)
        >
    hpx::future<void>
    define_spmd_block(std::string && name, std::size_t images_per_locality,
        F && f, Args && ... args)
    {
        using ftype = typename std::decay<F>::type;

        using helper_type =
            hpx::lcos::detail::spmd_block_helper_action<
               ftype, typename std::decay<Args>::type...>;

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
