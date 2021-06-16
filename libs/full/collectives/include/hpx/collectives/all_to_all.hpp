//  Copyright (c) 2019-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file all_to_all.hpp

#pragma once

#if defined(DOXYGEN)
// clang-format off
namespace hpx { namespace collectives {

    /// AllToAll a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the all_to_all operation
    /// \param  local_result The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the all_to_all operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the all_to_all operation on the
    ///                     given base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \params root_site   The site that is responsible for creating the
    ///                     all_to_all support object. This value is optional
    ///                     and defaults to '0' (zero).
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             values send by all participating sites. It will become
    ///             ready once the all_to_all operation has been completed.
    ///
    template <typename T>
    hpx::future<std::vector<std::decay_t<T>>>
    all_to_all(char const* basename, T&& result,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg());

    /// AllToAll a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  comm        A communicator object returned from \a create_reducer
    /// \param  local_result The value to transmit to all
    ///                     participating sites from this call site.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             values send by all participating sites. It will become
    ///             ready once the all_to_all operation has been completed.
    ///
    template <typename T>
    hpx::future<std::vector<std::decay_t<T>>>
    all_to_all(communicator comm, T&& result,
        this_site_arg this_site = this_site_arg());
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
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace traits {

    namespace communication {
        struct all_to_all_tag;
    }    // namespace communication

    ///////////////////////////////////////////////////////////////////////////
    // support for all_to_all
    template <typename Communicator>
    struct communication_operation<Communicator, communication::all_to_all_tag>
      : std::enable_shared_from_this<communication_operation<Communicator,
            communication::all_to_all_tag>>
    {
        explicit communication_operation(Communicator& comm)
          : communicator_(comm)
        {
        }

        template <typename Result, typename T>
        Result get(std::size_t which, std::vector<T>&& t)
        {
            using data_type = std::vector<T>;
            using mutex_type = typename Communicator::mutex_type;
            using lock_type = std::unique_lock<mutex_type>;

            auto this_ = this->shared_from_this();
            auto on_ready = [this_ = std::move(this_), which](
                                shared_future<void>&& f) -> data_type {
                HPX_UNUSED(this_);
                f.get();    // propagate any exceptions

                auto& communicator = this_->communicator_;

                lock_type l(communicator.mtx_);
                auto& data = communicator.template access_data<data_type>(l);

                // slice the overall data based on the locality id of the
                // requesting site
                std::vector<T> result;
                result.reserve(data.size());

                for (auto const& v : data)
                {
                    result.push_back(v[which]);
                }

                return result;
            };

            lock_type l(communicator_.mtx_);
            util::ignore_while_checking<lock_type> il(&l);

            hpx::future<data_type> f =
                communicator_.gate_.get_shared_future(l).then(
                    hpx::launch::sync, on_ready);

            communicator_.gate_.synchronize(1, l);

            auto& data = communicator_.template access_data<data_type>(l);
            data[which] = std::move(t);

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
    // all_to_all plain values
    template <typename T>
    hpx::future<std::vector<T>> all_to_all(communicator fid,
        std::vector<T>&& local_result,
        this_site_arg this_site = this_site_arg())
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(agas::get_locality_id());
        }

        auto all_to_all_data_direct =
            [local_result = std::move(local_result), this_site](
                communicator&& c) -> hpx::future<std::vector<T>> {
            using action_type = typename detail::communicator_server::
                template communication_get_action<
                    traits::communication::all_to_all_tag,
                    hpx::future<std::vector<T>>, std::vector<T>>;

            // make sure id is kept alive as long as the returned future,
            // explicitly unwrap returned future
            hpx::future<std::vector<T>> result =
                async(action_type(), c, this_site, std::move(local_result));

            traits::detail::get_shared_state(result)->set_on_completed(
                [client = std::move(c)]() { HPX_UNUSED(client); });

            return result;
        };

        return fid.then(hpx::launch::sync, std::move(all_to_all_data_direct));
    }

    template <typename T>
    hpx::future<std::vector<T>> all_to_all(char const* basename,
        std::vector<T>&& local_result,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg())
    {
        return all_to_all(create_communicator(basename, num_sites, this_site,
                              generation, root_site),
            std::move(local_result), this_site);
    }
}}    // namespace hpx::collectives

////////////////////////////////////////////////////////////////////////////////
// compatibility functions
namespace hpx { namespace lcos {

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::all_to_all is deprecated, use hpx::collectives::all_to_all "
        "instead")
    hpx::future<std::vector<T>> all_to_all(char const* basename,
        std::vector<T>&& local_result, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)
    {
        return hpx::collectives::all_to_all(basename, std::move(local_result),
            hpx::collectives::num_sites_arg(num_sites),
            hpx::collectives::this_site_arg(this_site),
            hpx::collectives::generation_arg(generation),
            hpx::collectives::root_site_arg(root_site));
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::all_to_all is deprecated, use hpx::collectives::all_to_all "
        "instead")
    hpx::future<std::vector<T>> all_to_all(char const* basename,
        hpx::future<std::vector<T>>&& local_result,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)
    {
        return local_result.then([=](hpx::future<T>&& f) {
            return hpx::collectives::all_to_all(basename, f.get(),
                hpx::collectives::num_sites_arg(num_sites),
                hpx::collectives::this_site_arg(this_site),
                hpx::collectives::generation_arg(generation),
                hpx::collectives::root_site_arg(root_site));
        });
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::all_to_all is deprecated, use hpx::collectives::all_to_all "
        "instead")
    hpx::future<std::vector<T>> all_to_all(hpx::collectives::communicator comm,
        std::vector<T>&& local_result, std::size_t this_site = std::size_t(-1))
    {
        return hpx::collectives::all_to_all(std::move(comm),
            std::move(local_result),
            hpx::collectives::this_site_arg(this_site));
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::all_to_all is deprecated, use hpx::collectives::all_to_all "
        "instead")
    hpx::future<std::vector<T>> all_to_all(hpx::collectives::communicator comm,
        hpx::future<std::vector<T>>&& local_result,
        std::size_t this_site = std::size_t(-1))
    {
        return local_result.then([=](hpx::future<T>&& f) mutable {
            hpx::collectives::all_to_all(std::move(comm), f.get(),
                hpx::collectives::this_site_arg(this_site));
        });
    }

    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::create_all_to_all is deprecated, use "
        "hpx::collectives::create_communicator instead")
    inline hpx::collectives::communicator create_all_to_all(
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
#define HPX_REGISTER_ALLTOALL_DECLARATION(...) /**/

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ALLTOALL(...)             /**/

#endif    // !HPX_COMPUTE_DEVICE_CODE
#endif    // DOXYGEN
