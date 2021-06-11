//  Copyright (c) 2019-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file reduce.hpp

#pragma once

#if defined(DOXYGEN)
// clang-format off
namespace hpx { namespace collectives {

    /// Reduce a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the all_reduce operation
    /// \param  local_result A value to reduce on the central reduction point
    ///                     from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the all_reduce operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the all_reduce operation on the
    ///                     given base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             values send by all participating sites. It will become
    ///             ready once the all_reduce operation has been completed.
    ///
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> reduce_here(char const* basename, T&& result,
        F&& op, num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg());

    /// Reduce a set of values from different call sites
    ///
    /// This function receives a set of values that are the result of applying
    /// a given operator on values supplied from all call sites operating on
    /// the given base name.
    ///
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  local_result A value to reduce on the root_site from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future holding a value calculated
    ///             based on the values send by all participating sites. It will
    ///             become ready once the all_reduce operation has been completed.
    ///
    template <typename T, typename F>
    hpx::future<decay_t<T>> reduce_here(communicator comm, T&& local_result,
        F&& op, this_site_arg this_site = this_site_arg());

    /// Reduce a given value at the given call site
    ///
    /// This function transmits the value given by \a result to a central reduce
    /// site (where the corresponding \a reduce_here is executed)
    ///
    /// \param  basename    The base name identifying the reduction operation
    /// \param  result      A future referring to the value to transmit to the
    ///                     central reduction point from this call site.
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the reduction operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the reduction operation on the given
    ///                     base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \param root_site    The sequence number of the central reduction point
    ///                     (usually the locality id). This value is optional
    ///                     and defaults to 0.
    ///
    /// \returns    This function returns a future<void>. It will become ready
    ///             once the reduction operation has been completed.
    ///
    template <typename T, typename F>
    hpx::future<void> reduce_there(char const* basename, T&& result,
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg());

    /// Reduce a given value at the given call site
    ///
    /// This function transmits the value given by \a result to a central reduce
    /// site (where the corresponding \a reduce_here is executed)
    ///
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  local_result A value to reduce on the central reduction point
    ///                     from this call site.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future holding a value calculated
    ///             based on the values send by all participating sites. It will
    ///             become ready once the all_reduce operation has been completed.
    ///
    template <typename T>
    hpx::future<void> reduce_there(communicator comm, T&& local_result,
        this_site_arg this_site = this_site_arg());
}}    // namespace hpx::collectives

// clang-format on
#else

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_distributed/async.hpp>
#include <hpx/collectives/argument_types.hpp>
#include <hpx/collectives/create_communicator.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/parallel/algorithms/reduce.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace traits {

    namespace communication {
        struct reduce_tag;
    }    // namespace communication

    ///////////////////////////////////////////////////////////////////////////
    // support for all_reduce
    template <typename Communicator>
    struct communication_operation<Communicator, communication::reduce_tag>
      : std::enable_shared_from_this<
            communication_operation<Communicator, communication::reduce_tag>>
    {
        explicit communication_operation(Communicator& comm)
          : communicator_(comm)
        {
        }

        template <typename Result, typename T, typename F>
        Result get(std::size_t which, T&& t, F&& op)
        {
            using arg_type = std::decay_t<T>;
            using mutex_type = typename Communicator::mutex_type;
            using lock_type = std::unique_lock<mutex_type>;

            auto this_ = this->shared_from_this();
            auto on_ready =
                [this_ = std::move(this_), op = std::forward<F>(op)](
                    hpx::shared_future<void> f) mutable -> arg_type {
                HPX_UNUSED(this_);
                f.get();    // propagate any exceptions

                auto& communicator = this_->communicator_;

                lock_type l(communicator.mtx_);
                util::ignore_while_checking<lock_type> il(&l);

                auto& data = communicator.template access_data<arg_type>(l);

                auto it = data.begin();
                return hpx::reduce(++it, data.end(), *data.begin(), op);
            };

            lock_type l(communicator_.mtx_);
            util::ignore_while_checking<lock_type> il(&l);

            hpx::future<arg_type> f =
                communicator_.gate_.get_shared_future(l).then(
                    hpx::launch::sync, std::move(on_ready));

            communicator_.gate_.synchronize(1, l);

            auto& data = communicator_.template access_data<arg_type>(l);
            data[which] = std::forward<T>(t);

            if (communicator_.gate_.set(which, std::move(l)))
            {
                l = lock_type(communicator_.mtx_);
                communicator_.invalidate_data(l);
            }

            return f;
        }

        template <typename Result, typename T>
        Result set(std::size_t which, T&& t)
        {
            using arg_type = std::decay_t<T>;
            using mutex_type = typename Communicator::mutex_type;
            using lock_type = std::unique_lock<mutex_type>;

            auto this_ = this->shared_from_this();
            auto on_ready = [this_ = std::move(this_)](
                                shared_future<void>&& f) {
                HPX_UNUSED(this_);
                f.get();    // propagate any exceptions
            };

            lock_type l(communicator_.mtx_);
            util::ignore_while_checking<lock_type> il(&l);

            hpx::future<void> f = communicator_.gate_.get_shared_future(l).then(
                hpx::launch::sync, std::move(on_ready));

            communicator_.gate_.synchronize(1, l);

            auto& data = communicator_.template access_data<arg_type>(l);
            data[which] = std::forward<T>(t);

            if (communicator_.gate_.set(which, std::move(l)))
            {
                l = lock_type(communicator_.mtx_);
                communicator_.invalidate_data(l);
            }

            return f;
        }

        Communicator& communicator_;
    };
}}    // namespace hpx::traits

namespace hpx { namespace collectives {

    ///////////////////////////////////////////////////////////////////////////
    // reduce plain values
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> reduce_here(communicator fid, T&& local_result,
        F&& op, this_site_arg this_site = this_site_arg())
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(agas::get_locality_id());
        }

        using arg_type = std::decay_t<T>;

        auto reduction_data_direct =
            [op = std::forward<F>(op),
                local_result = std::forward<T>(local_result),
                this_site](communicator&& c) mutable -> hpx::future<arg_type> {
            using func_type = std::decay_t<F>;
            using action_type = typename detail::communicator_server::
                template communication_get_action<
                    traits::communication::reduce_tag, hpx::future<arg_type>,
                    arg_type, func_type>;

            // make sure id is kept alive as long as the returned future,
            // explicitly unwrap returned future
            hpx::future<arg_type> result = async(action_type(), c, this_site,
                std::forward<T>(local_result), std::forward<F>(op));

            traits::detail::get_shared_state(result)->set_on_completed(
                [client = std::move(c)]() { HPX_UNUSED(client); });

            return result;
        };

        return fid.then(hpx::launch::sync, std::move(reduction_data_direct));
    }

    template <typename T, typename F>
    hpx::future<std::decay_t<T>> reduce_here(char const* basename, T&& result,
        F&& op, num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg())
    {
        return reduce_here(create_communicator(basename, num_sites, this_site,
                               generation, root_site_arg(this_site.this_site_)),
            std::forward<T>(result), std::forward<F>(op), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    // reduce plain values
    template <typename T>
    hpx::future<void> reduce_there(communicator fid, T&& local_result,
        this_site_arg this_site = this_site_arg())
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(agas::get_locality_id());
        }

        using arg_type = std::decay_t<T>;

        auto reduction_there_data_direct =
            [this_site](communicator&& c,
                arg_type&& local_result) -> hpx::future<void> {
            using action_type = typename detail::communicator_server::
                template communication_set_action<
                    traits::communication::reduce_tag, hpx::future<void>,
                    arg_type>;

            // make sure id is kept alive as long as the returned future,
            // explicitly unwrap returned future
            hpx::future<void> result = async(
                action_type(), c, this_site, std::forward<T>(local_result));

            traits::detail::get_shared_state(result)->set_on_completed(
                [client = std::move(c)]() { HPX_UNUSED(client); });

            return result;
        };

        return dataflow(std::move(reduction_there_data_direct), std::move(fid),
            std::forward<T>(local_result));
    }

    template <typename T>
    hpx::future<void> reduce_there(char const* basename, T&& local_result,
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg())
    {
        HPX_ASSERT(this_site != root_site);
        return reduce_there(create_communicator(basename, num_sites_arg(),
                                this_site, generation, root_site),
            std::forward<T>(local_result), this_site);
    }
}}    // namespace hpx::collectives

#endif    // !HPX_COMPUTE_DEVICE_CODE
#endif    // DOXYGEN
