//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_CUDA_CONCURRENT_EXECUTOR_HPP
#define HPX_COMPUTE_CUDA_CONCURRENT_EXECUTOR_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)

#include <hpx/compute/cuda/concurrent_executor_parameters.hpp>
#include <hpx/compute/cuda/default_executor.hpp>
#include <hpx/compute/cuda/target.hpp>
#include <hpx/compute/host/block_executor.hpp>
#include <hpx/compute/host/target.hpp>

#include <boost/atomic.hpp>

#include <vector>

namespace hpx { namespace compute { namespace cuda
{
    template <typename Executor =
        hpx::threads::executors::local_priority_queue_attached_executor>
    struct concurrent_executor : hpx::parallel::executor_tag
    {
    private:
        typedef host::block_executor<Executor> host_executor_type;
        typedef cuda::default_executor cuda_executor_type;

        typedef hpx::parallel::executor_traits<host_executor_type> host_executor_traits;
        typedef hpx::parallel::executor_traits<cuda_executor_type> cuda_executor_traits;

    public:
        // By default, this executor relies on a special executor parameters
        // implementation which knows about the specifics of creating the
        // bulk-shape ranges for the accelerator.
        typedef concurrent_executor_parameters executor_parameters_type;

        concurrent_executor(
            cuda::target const& cuda_target, std::vector<host::target> const& host_targets)
          : host_executor_(host_targets),
            current_(0)
        {
            std::size_t num_targets = host_targets.size();
            cuda_executors_.reserve(num_targets);
            for(std::size_t i = 0; i != num_targets; ++i)
            {
                cuda::target t(cuda_target.native_handle().get_device());
                t.native_handle().get_stream();
                cuda_executors_.emplace_back(t);
            }
        }

        concurrent_executor(concurrent_executor const& other)
          : host_executor_(other.host_executor_)
          , cuda_executors_(other.cuda_executors_)
          , current_(0)
        {}

        concurrent_executor(concurrent_executor&& other)
          : host_executor_(std::move(other.host_executor_))
          , cuda_executors_(std::move(other.cuda_executors_))
          , current_(other.current_.load())
        {}

        concurrent_executor& operator=(concurrent_executor const& other)
        {
            if(&other != this)
            {
                host_executor_ = other.host_executor_;
                cuda_executors_ = other.cuda_executors_;
                current_ = 0;
            }

            return *this;
        }

        concurrent_executor& operator=(concurrent_executor&& other)
        {
            if(&other != this)
            {
                host_executor_ = std::move(other.host_executor_);
                cuda_executors_ = std::move(other.cuda_executors_);
                current_ = other.current_.load();
            }

            return *this;
        }

        std::size_t processing_units_count() const
        {
            return cuda_executors_.size();
        }

        template <typename F, typename ... Ts>
        void apply_execute(F && f, Ts &&... ts)
        {
            std::size_t current = ++current_ % cuda_executors_.size();
            host_executor_traits::apply_execute(host_executor_,
                [this, current](F&& f, Ts&&... ts) mutable
                {
                    cuda_executor_traits::apply_execute(cuda_executors_[current],
                        std::forward<F>(f), std::forward<Ts>(ts)...);
                },
                std::forward<F>(f), std::forward<Ts>(ts)...
            );
        }

        template <typename F, typename ... Ts>
        hpx::future<void> async_execute(F && f, Ts &&... ts)
        {
            std::size_t current = ++current_ % cuda_executors_.size();
            return host_executor_traits::async_execute(host_executor_,
                [this, current](F&& f, Ts&&... ts) mutable
                {
                    cuda_executor_traits::apply_execute(cuda_executors_[current],
                        std::forward<F>(f), std::forward<Ts>(ts)...);

                    return cuda_executors_[current].target().get_future();
                },
                std::forward<F>(f), std::forward<Ts>(ts)...
            );
        }

        template <typename F, typename ... Ts>
        void execute(F && f, Ts &&... ts)
        {
            std::size_t current = ++current_ % cuda_executors_.size();
            host_executor_traits::execute(host_executor_,
                [this, current](F&& f, Ts&&... ts) mutable
                {
                    cuda_executor_traits::apply_execute(cuda_executors_[current],
                        std::forward<F>(f), std::forward<Ts>(ts)...);

                    cuda_executors_[current].target().synchronize();
                },
                std::forward<F>(f), std::forward<Ts>(ts)...
            );
        }

        template <typename F, typename Shape, typename ... Ts>
        std::vector<hpx::future<void> >
        bulk_async_execute(F && f, Shape const& shape, Ts &&... ts)
        {
// Before Boost V1.56 boost::size() does not respect the iterator category of
// its argument.
#if BOOST_VERSION < 105600
            std::size_t cnt = std::distance(boost::begin(shape), boost::end(shape));
#else
            std::size_t cnt = boost::size(shape);
#endif
            std::vector<hpx::future<void> > result;
            result.reserve(cnt);

            for (auto const& s: shape)
            {
                std::size_t current = ++current_ % cuda_executors_.size();
                result.push_back(host_executor_traits::async_execute(host_executor_,
                    [this, current, s](F&& f, Ts&&... ts) mutable
                    {
                        std::array<typename hpx::util::decay<decltype(s)>::type, 1> cuda_shape{{s}};
//                         return cuda_executor_traits::bulk_async_execute(cuda_executors_[current],
//                             std::forward<F>(f), cuda_shape, std::forward<Ts>(ts)...);
                        cuda_executors_[current].bulk_launch(std::forward<F>(f), cuda_shape, std::forward<Ts>(ts)...);
//                         return cuda_executors_[current].target().get_future();
                        cuda_executors_[current].target().synchronize();
                    },
                    std::forward<F>(f), std::forward<Ts>(ts)...));

            }
            return result;
        }

    private:
        host_executor_type host_executor_;
        std::vector<cuda_executor_type> cuda_executors_;
        boost::atomic<std::size_t> current_;
    };
}}}

#endif
#endif
