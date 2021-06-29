//  Copyright (c) 2014-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file scatter.hpp

#pragma once

#if defined(DOXYGEN)
// clang-format off
namespace hpx { namespace collectives {

    /// Scatter (receive) a set of values to different call sites
    ///
    /// This function receives an element of a set of values operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the scatter operation
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the scatter operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the scatter operation on the given
    ///                     base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \param root_site    The sequence number of the central scatter point
    ///                     (usually the locality id). This value is optional
    ///                     and defaults to 0.
    ///
    /// \returns    This function returns a future holding a the
    ///             scattered value. It will become ready once the scatter
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<T> scatter_from(char const* basename,
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg());

    /// Scatter (receive) a set of values to different call sites
    ///
    /// This function receives an element of a set of values operating on
    /// the given base name.
    ///
    /// \param  comm        A communicator object returned from \a create_reducer
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future holding a the
    ///             scattered value. It will become ready once the scatter
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<T> scatter_from(communicator comm,
        this_site_arg this_site = this_site_arg());

    /// Scatter (send) a part of the value set at the given call site
    ///
    /// This function transmits the value given by \a result to a central scatter
    /// site (where the corresponding \a scatter_from is executed)
    ///
    /// \param  basename    The base name identifying the scatter operation
    /// \param  result      The value to transmit to the central scatter point
    ///                     from this call site.
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the scatter operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the scatter operation on the given
    ///                     base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future holding a the
    ///             scattered value. It will become ready once the scatter
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<T> scatter_to(char const* basename,
        std::vector<T>&& result,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg());

    /// Scatter (send) a part of the value set at the given call site
    ///
    /// This function transmits the value given by \a result to a central scatter
    /// site (where the corresponding \a scatter_from is executed)
    ///
    /// \param  comm        A communicator object returned from \a create_reducer
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future holding a the
    ///             scattered value. It will become ready once the scatter
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<T> scatter_to(communicator comm,
        std::vector<T>&& result, this_site_arg this_site = this_site_arg());
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
    // support for scatter
    namespace communication {
        struct scatter_tag;
    }    // namespace communication

    template <typename Communicator>
    struct communication_operation<Communicator, communication::scatter_tag>
      : std::enable_shared_from_this<
            communication_operation<Communicator, communication::scatter_tag>>
    {
        explicit communication_operation(Communicator& comm)
          : communicator_(comm)
        {
        }

        template <typename Result>
        Result get(std::size_t which)
        {
            using arg_type = typename Result::result_type;
            using mutex_type = typename Communicator::mutex_type;
            using lock_type = std::unique_lock<mutex_type>;

            auto this_ = this->shared_from_this();
            auto on_ready = [this_ = std::move(this_), which](
                                shared_future<void>&& f) -> arg_type {
                HPX_UNUSED(this_);
                f.get();    // propagate any exceptions

                auto& communicator = this_->communicator_;

                lock_type l(communicator.mtx_);

                auto& data = communicator.template access_data<arg_type>(l);
                return std::move(data[which]);
            };

            lock_type l(communicator_.mtx_);
            util::ignore_while_checking<lock_type> il(&l);

            hpx::future<arg_type> f =
                communicator_.gate_.get_shared_future(l).then(
                    hpx::launch::sync, std::move(on_ready));

            communicator_.gate_.synchronize(1, l);

            if (communicator_.gate_.set(which, std::move(l)))
            {
                l = lock_type(communicator_.mtx_);
                communicator_.invalidate_data(l);
            }

            return f;
        }

        template <typename Result, typename T>
        Result set(std::size_t which, std::vector<T>&& t)
        {
            using mutex_type = typename Communicator::mutex_type;
            using lock_type = std::unique_lock<mutex_type>;

            auto this_ = this->shared_from_this();
            auto on_ready = [this_ = std::move(this_), which](
                                shared_future<void>&& f) -> T {
                HPX_UNUSED(this_);
                f.get();    // propagate any exceptions

                auto& communicator = this_->communicator_;

                lock_type l(communicator.mtx_);

                auto& data = communicator.template access_data<T>(l);
                return std::move(data[which]);
            };

            lock_type l(communicator_.mtx_);
            util::ignore_while_checking<lock_type> il(&l);

            hpx::future<T> f = communicator_.gate_.get_shared_future(l).then(
                hpx::launch::sync, std::move(on_ready));

            communicator_.gate_.synchronize(1, l);

            auto& data = communicator_.template access_data<T>(l);
            data = std::move(t);

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
    // destination site needs to be handled differently
    template <typename T>
    hpx::future<T> scatter_from(
        communicator fid, this_site_arg this_site = this_site_arg())
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(agas::get_locality_id());
        }

        auto scatter_data = [this_site](communicator&& c) -> hpx::future<T> {
            using action_type = typename detail::communicator_server::
                template communication_get_action<
                    traits::communication::scatter_tag, hpx::future<T>>;

            // make sure id is kept alive as long as the returned future,
            // explicitly unwrap returned future
            hpx::future<T> result = async(action_type(), c, this_site);

            traits::detail::get_shared_state(result)->set_on_completed(
                [client = std::move(c)]() { HPX_UNUSED(client); });

            return result;
        };

        return fid.then(hpx::launch::sync, std::move(scatter_data));
    }

    template <typename T>
    hpx::future<T> scatter_from(char const* basename,
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg())
    {
        HPX_ASSERT(this_site != root_site);
        return scatter_from<T>(create_communicator(basename, num_sites_arg(),
                                   this_site, generation, root_site),
            this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    // scatter plain values
    template <typename T>
    hpx::future<T> scatter_to(communicator fid, std::vector<T>&& local_result,
        this_site_arg this_site = this_site_arg())
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(agas::get_locality_id());
        }

        auto scatter_to_data_direct =
            [this_site](communicator&& c,
                std::vector<T>&& local_result) -> hpx::future<T> {
            using action_type = typename detail::communicator_server::
                template communication_set_action<
                    traits::communication::scatter_tag, hpx::future<T>,
                    std::vector<T>>;

            // make sure id is kept alive as long as the returned future,
            // explicitly unwrap returned future
            hpx::future<T> result =
                async(action_type(), c, this_site, std::move(local_result));

            traits::detail::get_shared_state(result)->set_on_completed(
                [client = std::move(c)]() { HPX_UNUSED(client); });

            return result;
        };

        return dataflow(std::move(scatter_to_data_direct), std::move(fid),
            std::move(local_result));
    }

    template <typename T>
    hpx::future<T> scatter_to(char const* basename,
        std::vector<T>&& local_result,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg())
    {
        return scatter_to(create_communicator(basename, num_sites, this_site,
                              generation, root_site_arg(this_site.this_site_)),
            std::move(local_result), this_site);
    }
}}    // namespace hpx::collectives

////////////////////////////////////////////////////////////////////////////////
// compatibility functions
namespace hpx { namespace lcos {

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::scatter_to is deprecated, use "
        "hpx::collectives::scatter_to instead")
    hpx::future<T> scatter_to(char const* basename,
        std::vector<T>&& local_result, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
    {
        return hpx::collectives::scatter_to(basename, std::move(local_result),
            hpx::collectives::num_sites_arg(num_sites),
            hpx::collectives::this_site_arg(this_site),
            hpx::collectives::generation_arg(generation));
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::scatter_to is deprecated, use "
        "hpx::collectives::scatter_to instead")
    hpx::future<T> scatter_to(char const* basename,
        hpx::future<std::vector<T>>&& local_result,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
    {
        return local_result.then([=](hpx::future<T>&& f) {
            return hpx::collectives::scatter_to(basename, f.get(),
                hpx::collectives::num_sites_arg(num_sites),
                hpx::collectives::this_site_arg(this_site),
                hpx::collectives::generation_arg(generation));
        });
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::scatter is deprecated, use hpx::collectives::scatter "
        "instead")
    hpx::future<T> scatter_to(hpx::collectives::communicator comm,
        std::vector<T>&& local_result, std::size_t this_site = std::size_t(-1))
    {
        return hpx::collectives::scatter_to(std::move(comm),
            std::move(local_result),
            hpx::collectives::this_site_arg(this_site));
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::scatter_to is deprecated, use "
        "hpx::collectives::scatter_to instead")
    hpx::future<T> scatter_to(hpx::collectives::communicator comm,
        hpx::future<std::vector<T>>&& local_result,
        std::size_t this_site = std::size_t(-1))
    {
        return local_result.then([=](hpx::future<T>&& f) mutable {
            hpx::collectives::scatter_to(std::move(comm), f.get(),
                hpx::collectives::this_site_arg(this_site));
        });
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::scatter_from is deprecated, use "
        "hpx::collectives::scatter_from instead")
    hpx::future<T> scatter_from(char const* basename,
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)
    {
        return hpx::collectives::scatter_from<T>(basename,
            hpx::collectives::this_site_arg(this_site),
            hpx::collectives::generation_arg(generation),
            hpx::collectives::root_site_arg(root_site));
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::scatter_from is deprecated, use "
        "hpx::collectives::scatter_from instead")
    hpx::future<T> scatter_from(hpx::collectives::communicator comm,
        std::size_t this_site = std::size_t(-1))
    {
        return hpx::collectives::scatter_from<T>(
            std::move(comm), hpx::collectives::this_site_arg(this_site));
    }

    HPX_DEPRECATED_V(1, 7,
        "hpx::lcos::create_scatter is deprecated, use "
        "hpx::collectives::create_communicator instead")
    inline hpx::collectives::communicator create_scatterer(char const* basename,
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
#define HPX_REGISTER_SCATTER_DECLARATION(...) /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_SCATTER(...)             /**/

#endif    // !HPX_COMPUTE_DEVICE_CODE
#endif    // DOXYGEN
