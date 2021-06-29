//  Copyright (c) 2019-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file all_reduce.hpp

#pragma once

#if defined(DOXYGEN)
// clang-format off
namespace hpx { namespace collectives {

    /// AllReduce a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the all_reduce operation
    /// \param  local_result The value to transmit to all
    ///                     participating sites from this call site.
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
    /// \params root_site   The site that is responsible for creating the
    ///                     all_reduce support object. This value is optional
    ///                     and defaults to '0' (zero).
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             values send by all participating sites. It will become
    ///             ready once the all_reduce operation has been completed.
    ///
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> all_reduce(char const* basename, T&& result,
        F&& op, num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg());

    /// AllReduce a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  comm        A communicator object returned from \a create_reducer
    /// \param  local_result The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             values send by all participating sites. It will become
    ///             ready once the all_reduce operation has been completed.
    ///
    template <typename T, typename F>
    hpx::future<std::decay_t<T>>
    all_reduce(communicator comm,
        T&& result, F&& op, this_site_arg this_site = this_site_arg());
}}    // namespace hpx::collectives

// clang-format on
#else

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

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
        struct all_reduce_tag;
    }    // namespace communication

    ///////////////////////////////////////////////////////////////////////////
    // support for all_reduce
    template <typename Communicator>
    struct communication_operation<Communicator, communication::all_reduce_tag>
      : std::enable_shared_from_this<communication_operation<Communicator,
            communication::all_reduce_tag>>
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
                if (!communicator.data_available_)
                {
                    // compute reduction result only once
                    auto it = data.begin();
                    data[0] = hpx::reduce(++it, data.end(), *data.begin(), op);
                    communicator.data_available_ = true;
                }
                return data[0];
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

        Communicator& communicator_;
    };
}}    // namespace hpx::traits

namespace hpx { namespace collectives {

    ////////////////////////////////////////////////////////////////////////////
    // all_reduce plain values
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> all_reduce(communicator fid, T&& local_result,
        F&& op, this_site_arg this_site = this_site_arg())
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(agas::get_locality_id());
        }

        using arg_type = std::decay_t<T>;

        auto all_reduce_data_direct =
            [op = std::forward<F>(op),
                local_result = std::forward<T>(local_result),
                this_site](communicator&& c) mutable -> hpx::future<arg_type> {
            using func_type = std::decay_t<F>;
            using action_type = typename detail::communicator_server::
                template communication_get_action<
                    traits::communication::all_reduce_tag,
                    hpx::future<arg_type>, arg_type, func_type>;

            // make sure id is kept alive as long as the returned future,
            // explicitly unwrap returned future
            hpx::future<arg_type> result = async(action_type(), c, this_site,
                std::forward<T>(local_result), std::forward<F>(op));

            traits::detail::get_shared_state(result)->set_on_completed(
                [client = std::move(c)]() { HPX_UNUSED(client); });

            return result;
        };

        return fid.then(hpx::launch::sync, std::move(all_reduce_data_direct));
    }

    template <typename T, typename F>
    hpx::future<std::decay_t<T>> all_reduce(char const* basename,
        T&& local_result, F&& op, num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg())
    {
        return all_reduce(create_communicator(basename, num_sites, this_site,
                              generation, root_site),
            std::forward<T>(local_result), std::forward<F>(op), this_site);
    }
}}    // namespace hpx::collectives

////////////////////////////////////////////////////////////////////////////////
// compatibility functions
namespace hpx { namespace lcos {

    template <typename T, typename F>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::all_reduce is deprecated, use hpx::collectives::all_reduce "
        "instead")
    hpx::future<std::decay_t<T>> all_reduce(char const* basename,
        T&& local_result, F&& op, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)
    {
        return hpx::collectives::all_reduce(basename,
            std::forward<T>(local_result), std::forward<F>(op),
            hpx::collectives::num_sites_arg(num_sites),
            hpx::collectives::this_site_arg(this_site),
            hpx::collectives::generation_arg(generation),
            hpx::collectives::root_site_arg(root_site));
    }

    template <typename T, typename F>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::all_reduce is deprecated, use hpx::collectives::all_reduce "
        "instead")
    hpx::future<T> all_reduce(char const* basename,
        hpx::future<T>&& local_result, F&& op,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)
    {
        return local_result.then([=](hpx::future<T>&& f) mutable {
            // silence bogus clang-tidy error
            return hpx::collectives::all_reduce(basename, f.get(),
                // NOLINTNEXTLINE(bugprone-move-forwarding-reference)
                std::move(op), hpx::collectives::num_sites_arg(num_sites),
                hpx::collectives::this_site_arg(this_site),
                hpx::collectives::generation_arg(generation),
                hpx::collectives::root_site_arg(root_site));
        });
    }

    template <typename T, typename F>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::all_reduce is deprecated, use hpx::collectives::all_reduce "
        "instead")
    hpx::future<std::decay_t<T>> all_reduce(hpx::collectives::communicator comm,
        T&& local_result, F&& op, std::size_t this_site = std::size_t(-1))
    {
        return hpx::collectives::all_reduce(std::move(comm),
            std::forward<T>(local_result), std::forward<F>(op),
            hpx::collectives::this_site_arg(this_site));
    }

    template <typename T, typename F>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::all_reduce is deprecated, use hpx::collectives::all_reduce "
        "instead")
    hpx::future<T> all_reduce(hpx::collectives::communicator comm,
        hpx::future<T>&& local_result, F&& op,
        std::size_t this_site = std::size_t(-1))
    {
        return local_result.then([=](hpx::future<T>&& f) mutable {
            // silence bogus clang-tidy error
            hpx::collectives::all_reduce(std::move(comm), f.get(),
                // NOLINTNEXTLINE(bugprone-move-forwarding-reference)
                std::move(op), hpx::collectives::this_site_arg(this_site));
        });
    }

    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::create_all_reduce is deprecated, use "
        "hpx::collectives::create_communicator instead")
    inline hpx::collectives::communicator create_all_reduce(
        char const* basename, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
    {
        return hpx::collectives::create_communicator(basename,
            hpx::collectives::num_sites_arg(num_sites),
            hpx::collectives::this_site_arg(this_site),
            hpx::collectives::generation_arg(generation));
    }
}}    // namespace hpx::lcos

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ALLREDUCE_DECLARATION(...) /**/

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ALLREDUCE(...)             /**/

#endif    // !HPX_COMPUTE_DEVICE_CODE
#endif    // DOXYGEN
