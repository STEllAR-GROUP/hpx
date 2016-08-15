//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_HOST_BLOCK_EXECUTOR_HPP
#define HPX_COMPUTE_HOST_BLOCK_EXECUTOR_HPP

#include <hpx/config.hpp>
#include <hpx/compute/host/target.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/runtime/threads/executors/thread_pool_attached_executors.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/unwrapped.hpp>

#include <boost/atomic.hpp>
#include <boost/range/iterator_range_core.hpp>

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

namespace hpx { namespace compute { namespace host
{
    /// The block executor can be used to build NUMA aware programs.
    /// It will distribute work evenly accross the passed targets
    ///
    /// \tparam Executor The underlying executor to use
    template <typename Executor =
        hpx::threads::executors::local_priority_queue_attached_executor>
    struct block_executor : hpx::parallel::executor_tag
    {
    private:
        typedef hpx::parallel::executor_traits<Executor> executor_traits;

    public:
        typedef hpx::parallel::static_chunk_size executor_parameters_type;

        block_executor(std::vector<host::target> const& targets)
          : targets_(targets)
          , current_(0)
        {
            init_executors();
        }

        block_executor(std::vector<host::target>&& targets)
          : targets_(std::move(targets))
          , current_(0)
        {
            init_executors();
        }

        block_executor(block_executor const& other)
          : targets_(other.targets_)
          , current_(0)
          , executors_(other.executors_)
        {}

        block_executor(block_executor&& other)
          : targets_(std::move(other.targets_))
          , current_(other.current_.load())
          , executors_(std::move(other.executors_))
        {}

        block_executor& operator=(block_executor const& other)
        {
            if (&other != this)
            {
                targets_ = other.targets_;
                current_ = 0;
                executors_ = other.executors_;
            }
            return *this;
        }

        block_executor& operator=(block_executor && other)
        {
            if (&other != this)
            {
                targets_ = std::move(other.targets_);
                current_ = other.current_.load();
                executors_ = std::move(other.executors_);
            }
            return *this;
        }

        template <typename F, typename ... Ts>
        void apply_execute(F && f, Ts &&... ts)
        {
            executor_traits::apply_execute(
                executors_[current_],
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename ... Ts>
        hpx::future<
            typename hpx::util::detail::deferred_result_of<F(Ts&&...)>::type>
        async_execute(F && f, Ts &&... ts)
        {
            std::size_t current = ++current_ % executors_.size();
            return executor_traits::async_execute(executors_[current],
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename ... Ts>
        typename hpx::util::detail::deferred_result_of<F(Ts&&...)>::type
        execute(F && f, Ts &&... ts)
        {
            std::size_t current = ++current_ % executors_.size();
            return executor_traits::execute(executors_[current],
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename Shape, typename ... Ts>
        std::vector<hpx::future<
            typename hpx::parallel::v3::detail::bulk_async_execute_result<
                F, Shape, Ts...
            >::type>
        >
        bulk_async_execute(F && f, Shape const& shape, Ts &&... ts)
        {
            std::vector<hpx::future<
                typename hpx::parallel::v3::detail::bulk_async_execute_result<
                        F, Shape, Ts...
                    >::type
            > > results;
// Before Boost V1.56 boost::size() does not respect the iterator category of
// its argument.
#if BOOST_VERSION < 105600
            std::size_t cnt = std::distance(boost::begin(shape), boost::end(shape));
#else
            std::size_t cnt = boost::size(shape);
#endif
            std::size_t part_size = cnt / executors_.size();

            results.reserve(cnt);

            try {
                auto begin = boost::begin(shape);
                for (std::size_t i = 0; i != executors_.size(); ++i)
                {
                    auto part_end = begin;
                    std::advance(part_end, part_size);
                    auto futures =
                        executor_traits::bulk_async_execute(
                            executors_[i],
                            std::forward<F>(f),
                            boost::make_iterator_range(begin, part_end),
                            std::forward<Ts>(ts)...);
                    results.insert(
                        results.end(),
                        std::make_move_iterator(futures.begin()),
                        std::make_move_iterator(futures.end()));
                    begin = part_end;
                }
                return results;
            }
            catch (std::bad_alloc const& ba) {
                boost::throw_exception(ba);
            }
            catch (...) {
                boost::throw_exception(
                    exception_list(boost::current_exception())
                );
            }
        }

        template <typename F, typename Shape, typename ... Ts>
        typename hpx::parallel::v3::detail::bulk_execute_result<
            F, Shape, Ts...
        >::type
        bulk_execute(F && f, Shape const& shape, Ts &&... ts)
        {
            typename hpx::parallel::v3::detail::bulk_execute_result<
                    F, Shape, Ts...
                >::type results;
// Before Boost V1.56 boost::size() does not respect the iterator category of
// its argument.
#if BOOST_VERSION < 105600
            std::size_t cnt = std::distance(boost::begin(shape), boost::end(shape));
#else
            std::size_t cnt = boost::size(shape);
#endif
            std::size_t part_size = cnt / executors_.size();

            results.reserve(cnt);

            try {
                auto begin = boost::begin(shape);
                for (std::size_t i = 0; i != executors_.size(); ++i)
                {
                    auto part_end = begin;
                    std::advance(part_end, part_size);
                    auto part_results =
                        executor_traits::bulk_execute(
                            executors_[i],
                            std::forward<F>(f),
                            boost::make_iterator_range(begin, part_end),
                            std::forward<Ts>(ts)...);
                    results.insert(
                        results.end(),
                        std::make_move_iterator(part_results.begin()),
                        std::make_move_iterator(part_results.end()));
                    begin = part_end;
                }
                return results;
            }
            catch (std::bad_alloc const& ba) {
                boost::throw_exception(ba);
            }
            catch (...) {
                boost::throw_exception(
                    exception_list(boost::current_exception())
                );
            }
        }

        std::vector<host::target> const& targets() const
        {
            return targets_;
        }

    private:
        void init_executors()
        {
            executors_.reserve(targets_.size());
            for(auto const & tgt : targets_)
            {
                auto num_pus = tgt.num_pus();
                executors_.emplace_back(num_pus.first, num_pus.second);
            }
        }
        std::vector<host::target> targets_;
        boost::atomic<std::size_t> current_;
        std::vector<Executor> executors_;
    };
}}}

#endif
