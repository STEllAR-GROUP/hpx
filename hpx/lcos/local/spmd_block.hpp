//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_SPMD_BLOCK_HPP)
#define HPX_LCOS_LOCAL_SPMD_BLOCK_HPP

#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/barrier.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/traits/is_execution_policy.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/first_argument.hpp>

#include <boost/range/irange.hpp>

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace lcos { namespace local
{
    /// The class spmd_block defines an interface for launching
    /// multiple images while giving handles to each image to interact with
    /// the remaining images. The \a define_spmd_block function templates create
    /// multiple images of a user-defined function (or lambda) and launches them
    /// in a possibly separate thread. A temporary spmd block object is created
    /// and diffused to each image. The constraint for the function (or lambda)
    /// given to the define_spmd_block function is to accept a spmd_block as
    /// first parameter.
    struct spmd_block
    {
    private:
        using barrier_type = hpx::lcos::local::barrier;
        using table_type =
            std::map<std::set<std::size_t>,std::shared_ptr<barrier_type>>;
        using mutex_type = hpx::lcos::local::mutex;

    public:
        explicit spmd_block(std::size_t num_images, std::size_t image_id,
            barrier_type & barrier,table_type & barriers, mutex_type & mtx)
        : num_images_(num_images), image_id_(image_id), barrier_(barrier),
            barriers_(barriers), mtx_(mtx)
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

        void sync_images(std::set<std::size_t> const & images) const
        {
            using lock_type = std::lock_guard<mutex_type>;

            table_type & brs = barriers_.get();
            typename table_type::iterator it;

            // Critical section
            {
                lock_type lk(mtx_);
                it = brs.find(images);

                if(it == brs.end())
                {
                    it = brs.insert({images,
                        std::make_shared<barrier_type>(images.size())}).first;
                }
            }

            if( images.find(image_id_) != images.end() )
            {
                it->second->wait();
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
        sync_images(I ... i) const
        {
            std::set<std::size_t> images = {(std::size_t)i...};
            sync_images(images);
        }

    private:
        std::size_t num_images_;
        std::size_t image_id_;
        mutable std::reference_wrapper<barrier_type> barrier_;
        mutable std::reference_wrapper<table_type> barriers_;
        mutable std::reference_wrapper<mutex_type> mtx_;
    };

    namespace detail
    {
        template <typename F>
        struct spmd_block_helper
        {
        private:
            using barrier_type = hpx::lcos::local::barrier;
            using table_type =
                std::map<std::set<std::size_t>,std::shared_ptr<barrier_type>>;
            using mutex_type = hpx::lcos::local::mutex;

        public:
            std::shared_ptr<barrier_type> barrier_;
            std::shared_ptr<table_type> barriers_;
            std::shared_ptr<mutex_type> mtx_;
            typename std::decay<F>::type f_;
            std::size_t num_images_;

            template <typename ... Ts>
            void operator()(std::size_t image_id, Ts && ... ts) const
            {
                spmd_block block(num_images_, image_id, *barrier_,
                    *barriers_, *mtx_);
                hpx::util::invoke(
                    f_, std::move(block), std::forward<Ts>(ts)...);
            }
        };
    }

    // Asynchronous version
    template <typename ExPolicy, typename F, typename ... Args,
        typename = typename std::enable_if<
            hpx::parallel::v1::is_async_execution_policy<ExPolicy>::value>::type
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
            typename hpx::util::first_argument<ftype>::type;
        using executor_type =
            typename hpx::util::decay<ExPolicy>::type::executor_type;

        using barrier_type = hpx::lcos::local::barrier;
        using table_type =
            std::map<std::set<std::size_t>,std::shared_ptr<barrier_type>>;
        using mutex_type = hpx::lcos::local::mutex;

        static_assert(std::is_same<spmd_block, first_type>::value,
            "define_spmd_block() needs a function or lambda that " \
            "has at least a local spmd_block as 1st argument");

        std::shared_ptr<barrier_type> barrier
            = std::make_shared<barrier_type>(num_images);
        std::shared_ptr<table_type> barriers
            = std::make_shared<table_type>();
        std::shared_ptr<mutex_type> mtx
            = std::make_shared<mutex_type>();

        return
            hpx::parallel::executor_traits<
                    typename std::decay<executor_type>::type
                >::bulk_async_execute(
                    policy.executor(),
                    detail::spmd_block_helper<F>{
                        barrier,barriers,mtx,std::forward<F>(f),num_images
                    },
                    boost::irange(std::size_t(0), num_images),
                        std::forward<Args>(args)...);
    }

    // Synchronous version
    template <typename ExPolicy, typename F, typename ... Args,
        typename = typename std::enable_if<
            !hpx::parallel::v1::is_async_execution_policy<ExPolicy>::value>::type
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
            typename hpx::util::first_argument<ftype>::type;
        using executor_type =
            typename hpx::util::decay<ExPolicy>::type::executor_type;

        using barrier_type = hpx::lcos::local::barrier;
        using table_type =
            std::map<std::set<std::size_t>,std::shared_ptr<barrier_type>>;
        using mutex_type = hpx::lcos::local::mutex;

        static_assert(std::is_same<spmd_block, first_type>::value,
            "define_spmd_block() needs a lambda that " \
            "has at least a spmd_block as 1st argument");

        std::shared_ptr<barrier_type> barrier
            = std::make_shared<barrier_type>(num_images);
        std::shared_ptr<table_type> barriers
            = std::make_shared<table_type>();
        std::shared_ptr<mutex_type> mtx
            = std::make_shared<mutex_type>();

        hpx::parallel::executor_traits<
                typename std::decay<executor_type>::type
            >::bulk_execute(
                policy.executor(),
                detail::spmd_block_helper<F>{
                    barrier,barriers,mtx,std::forward<F>(f),num_images
                },
                boost::irange(std::size_t(0), num_images),
                    std::forward<Args>(args)...);
    }

    template <typename F, typename ... Args>
    void define_spmd_block(std::size_t num_images, F && f, Args && ... args)
    {
        define_spmd_block(parallel::execution::par,
            num_images, std::forward<F>(f), std::forward<Args>(args)...);
    }
}}}

#endif
