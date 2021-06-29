//  Copyright (c) 2014-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file gather.hpp

#pragma once

#if defined(DOXYGEN)
// clang-format off
namespace hpx { namespace collectives {

    /// Gather a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the gather operation
    /// \param  result      The value to transmit to the central gather point
    ///                     from this call site.
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the gather operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the gather operation on the given
    ///                     base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             gathered values. It will become ready once the gather
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<std::vector<decay_t<T>>> gather_here(
        char const* basename, T&& result,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg());

    /// Gather a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  comm        A communicator object returned from \a create_reducer
    /// \param  result      The value to transmit to the central gather point
    ///                     from this call site.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             gathered values. It will become ready once the gather
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<std::vector<decay_t<T>>> gather_here(
        communicator comm, T&& result, this_site_arg this_site = this_site_arg());

    /// Gather a given value at the given call site
    ///
    /// This function transmits the value given by \a result to a central gather
    /// site (where the corresponding \a gather_here is executed)
    ///
    /// \param  basename    The base name identifying the gather operation
    /// \param  result      The value to transmit to the central gather point
    ///                     from this call site.
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the gather operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the gather operation on the given
    ///                     base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \param root_site    The sequence number of the central gather point
    ///                     (usually the locality id). This value is optional
    ///                     and defaults to 0.
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             gathered values. It will become ready once the gather
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<std::vector<decay_t<T>>>
    gather_there(char const* basename, T&& result,
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg());

    /// Gather a given value at the given call site
    ///
    /// This function transmits the value given by \a result to a central gather
    /// site (where the corresponding \a gather_here is executed)
    ///
    /// \param  comm        A communicator object returned from \a create_reducer
    /// \param  result      The value to transmit to the central gather point
    ///                     from this call site.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             gathered values. It will become ready once the gather
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<std::vector<decay_t<T>>>
    gather_there(communicator comm, T&& result,
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
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    // support for gather
    namespace communication {
        struct gather_tag;
    }    // namespace communication

    template <typename Communicator>
    struct communication_operation<Communicator, communication::gather_tag>
      : std::enable_shared_from_this<
            communication_operation<Communicator, communication::gather_tag>>
    {
        explicit communication_operation(Communicator& comm)
          : communicator_(comm)
        {
        }

        template <typename Result, typename T>
        Result get(std::size_t which, T&& t)
        {
            using arg_type = std::decay_t<T>;
            using data_type = std::vector<arg_type>;
            using mutex_type = typename Communicator::mutex_type;
            using lock_type = std::unique_lock<mutex_type>;

            auto this_ = this->shared_from_this();
            auto on_ready = [this_ = std::move(this_)](
                                shared_future<void>&& f) -> data_type {
                HPX_UNUSED(this_);
                f.get();    // propagate any exceptions

                auto& communicator = this_->communicator_;

                lock_type l(communicator.mtx_);
                return communicator.template access_data<arg_type>(l);
            };

            lock_type l(communicator_.mtx_);
            util::ignore_while_checking<lock_type> il(&l);

            hpx::future<data_type> f =
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
    // gather plain values
    template <typename T>
    hpx::future<std::vector<std::decay_t<T>>> gather_here(communicator fid,
        T&& local_result, this_site_arg this_site = this_site_arg())
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(agas::get_locality_id());
        }

        using arg_type = std::decay_t<T>;

        auto gather_data_direct =
            [this_site](communicator&& c,
                arg_type&& local_result) -> hpx::future<std::vector<arg_type>> {
            using action_type = typename detail::communicator_server::
                template communication_get_action<
                    traits::communication::gather_tag,
                    hpx::future<std::vector<arg_type>>, arg_type>;

            // make sure id is kept alive as long as the returned future,
            // explicitly unwrap returned future
            hpx::future<std::vector<arg_type>> result =
                async(action_type(), c, this_site, std::move(local_result));

            traits::detail::get_shared_state(result)->set_on_completed(
                [client = std::move(c)]() { HPX_UNUSED(client); });

            return result;
        };

        return dataflow(hpx::launch::sync, std::move(gather_data_direct),
            std::move(fid), std::forward<T>(local_result));
    }

    template <typename T>
    hpx::future<std::vector<std::decay_t<T>>> gather_here(char const* basename,
        T&& result, num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg())
    {
        return gather_here(create_communicator(basename, num_sites, this_site,
                               generation, root_site_arg(this_site.this_site_)),
            std::forward<T>(result), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    // gather plain values
    template <typename T>
    hpx::future<void> gather_there(communicator fid, T&& local_result,
        this_site_arg this_site = this_site_arg())
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(agas::get_locality_id());
        }

        using arg_type = std::decay_t<T>;

        auto gather_there_data_direct =
            [this_site](communicator&& c,
                arg_type&& local_result) -> hpx::future<void> {
            using action_type = typename detail::communicator_server::
                template communication_set_action<
                    traits::communication::gather_tag, hpx::future<void>,
                    arg_type>;

            // make sure id is kept alive as long as the returned future,
            // explicitly unwrap returned future
            hpx::future<void> result =
                async(action_type(), c, this_site, std::move(local_result));

            traits::detail::get_shared_state(result)->set_on_completed(
                [client = std::move(c)]() { HPX_UNUSED(client); });

            return result;
        };

        return dataflow(std::move(gather_there_data_direct), std::move(fid),
            std::forward<T>(local_result));
    }

    template <typename T>
    hpx::future<void> gather_there(char const* basename, T&& local_result,
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg())
    {
        HPX_ASSERT(this_site != root_site);
        return gather_there(create_communicator(basename, num_sites_arg(),
                                this_site, generation, root_site),
            std::forward<T>(local_result), this_site);
    }
}}    // namespace hpx::collectives

////////////////////////////////////////////////////////////////////////////////
// compatibility functions
namespace hpx { namespace lcos {

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::gather_here is deprecated, use "
        "hpx::collectives::gather_here instead")
    hpx::future<std::vector<typename std::decay<T>::type>> gather_here(
        char const* basename, T&& local_result,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
    {
        return hpx::collectives::gather_here(basename,
            std::forward<T>(local_result),
            hpx::collectives::num_sites_arg(num_sites),
            hpx::collectives::this_site_arg(this_site),
            hpx::collectives::generation_arg(generation));
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::gather_here is deprecated, use "
        "hpx::collectives::gather_here instead")
    hpx::future<std::vector<T>> gather_here(char const* basename,
        hpx::future<T>&& local_result, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
    {
        return local_result.then([=](hpx::future<T>&& f) {
            return hpx::collectives::gather_here(basename, f.get(),
                hpx::collectives::num_sites_arg(num_sites),
                hpx::collectives::this_site_arg(this_site),
                hpx::collectives::generation_arg(generation));
        });
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::gather is deprecated, use hpx::collectives::gather "
        "instead")
    hpx::future<std::vector<std::decay_t<T>>> gather_here(
        hpx::collectives::communicator comm, T&& local_result,
        std::size_t this_site = std::size_t(-1))
    {
        return hpx::collectives::gather_here(std::move(comm),
            std::forward<T>(local_result),
            hpx::collectives::this_site_arg(this_site));
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::gather_here is deprecated, use "
        "hpx::collectives::gather_here instead")
    hpx::future<std::vector<T>> gather_here(hpx::collectives::communicator comm,
        hpx::future<T>&& local_result, std::size_t this_site = std::size_t(-1))
    {
        return local_result.then([=](hpx::future<T>&& f) mutable {
            hpx::collectives::gather_here(std::move(comm), f.get(),
                hpx::collectives::this_site_arg(this_site));
        });
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::gather_there is deprecated, use "
        "hpx::collectives::gather_there instead")
    hpx::future<void> gather_there(char const* basename, T&& local_result,
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)
    {
        return hpx::collectives::gather_there(basename,
            std::forward<T>(local_result),
            hpx::collectives::this_site_arg(this_site),
            hpx::collectives::generation_arg(generation),
            hpx::collectives::root_site_arg(root_site));
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::gather_there is deprecated, use "
        "hpx::collectives::gather_there instead")
    hpx::future<void> gather_there(char const* basename,
        hpx::future<T>&& local_result, std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)
    {
        return local_result.then([=](hpx::future<T>&& f) {
            return hpx::collectives::gather_there(basename, f.get(),
                hpx::collectives::this_site_arg(this_site),
                hpx::collectives::generation_arg(generation),
                hpx::collectives::root_site_arg(root_site));
        });
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::gather_there is deprecated, use "
        "hpx::collectives::gather_there instead")
    hpx::future<void> gather_there(hpx::collectives::communicator comm,
        T&& local_result, std::size_t this_site = std::size_t(-1))
    {
        return hpx::collectives::gather_there(std::move(comm),
            std::forward<T>(local_result),
            hpx::collectives::this_site_arg(this_site));
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::gather_there is deprecated, use "
        "hpx::collectives::gather_there instead")
    hpx::future<void> gather_there(hpx::collectives::communicator comm,
        hpx::future<T>&& local_result, std::size_t this_site = std::size_t(-1))
    {
        return local_result.then([=](hpx::future<T>&& f) {
            return hpx::collectives::gather_there(std::move(comm), f.get(),
                hpx::collectives::this_site_arg(this_site));
        });
    }

    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::create_gatherer is deprecated, use "
        "hpx::collectives::create_communicator instead")
    inline hpx::collectives::communicator create_gatherer(char const* basename,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
    {
        return hpx::collectives::create_communicator(basename,
            hpx::collectives::num_sites_arg(num_sites),
            hpx::collectives::this_site_arg(this_site),
            hpx::collectives::generation_arg(generation));
    }
}}    // namespace hpx::lcos

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_GATHER_DECLARATION(...) /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_GATHER(...)             /**/

#endif    // !HPX_COMPUTE_DEVICE_CODE
#endif    // DOXYGEN
