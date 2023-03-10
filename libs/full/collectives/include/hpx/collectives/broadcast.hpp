//  Copyright (c) 2020-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file broadcast.hpp

#pragma once

#if defined(DOXYGEN)
// clang-format off
namespace hpx { namespace collectives {

    /// Broadcast a value to different call sites
    ///
    /// This function sends a set of values to all call sites operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the broadcast operation
    /// \param  local_result A value to transmit to all
    ///                     participating sites from this call site.
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the broadcast operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the broadcast operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future that will become
    ///             ready once the broadcast operation has been completed.
    ///
    template <typename T>
    hpx::future<void> broadcast_to(char const* basename, T&& local_result,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg());

    /// Broadcast a value to different call sites
    ///
    /// This function sends a set of values to all call sites operating on
    /// the given base name.
    ///
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  local_result A value to transmit to all
    ///                     participating sites from this call site.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the broadcast operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the broadcast operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    ///
    /// \note       The generation values from corresponding \a broadcast_to and
    ///             \a broadcast_from have to match.
    ///
    /// \returns    This function returns a future that will become
    ///             ready once the broadcast operation has been completed.
    ///
    template <typename T>
    hpx::future<void> broadcast_to(communicator comm,
        T&& local_result, this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg());

    /// Broadcast a value to different call sites
    ///
    /// This function sends a set of values to all call sites operating on
    /// the given base name.
    ///
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  local_result A value to transmit to all
    ///                     participating sites from this call site.
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the broadcast operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the broadcast operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \note       The generation values from corresponding \a broadcast_to and
    ///             \a broadcast_from have to match.
    ///
    /// \returns    This function returns a future that will become
    ///             ready once the broadcast operation has been completed.
    ///
    template <typename T>
    hpx::future<void> broadcast_to(communicator comm,
        generation_arg generation,
        T&& local_result, this_site_arg this_site = this_site_arg());

    /// Receive a value that was broadcast to different call sites
    ///
    /// This function sends a set of values to all call sites operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the broadcast operation
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the broadcast operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the broadcast operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future holding the value that was
    ///             sent to all participating sites. It will become
    ///             ready once the broadcast operation has been completed.
    ///
    template <typename T>
    hpx::future<T> broadcast_from(char const* basename,
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg());

    /// Receive a value that was broadcast to different call sites
    ///
    /// This function sends a set of values to all call sites operating on
    /// the given base name.
    ///
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the broadcast operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the broadcast operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    ///
    /// \note       The generation values from corresponding \a broadcast_to and
    ///             \a broadcast_from have to match.
    ///
    /// \returns    This function returns a future holding the value that was
    ///             sent to all participating sites. It will become
    ///             ready once the broadcast operation has been completed.
    ///
    template <typename T>
    hpx::future<T> broadcast_from(communicator comm,
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg());

    /// Receive a value that was broadcast to different call sites
    ///
    /// This function sends a set of values to all call sites operating on
    /// the given base name.
    ///
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the broadcast operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the broadcast operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \note       The generation values from corresponding \a broadcast_to and
    ///             \a broadcast_from have to match.
    ///
    /// \returns    This function returns a future holding the value that was
    ///             sent to all participating sites. It will become
    ///             ready once the broadcast operation has been completed.
    ///
    template <typename T>
    hpx::future<T> broadcast_from(communicator comm,
        generation_arg generation,
        this_site_arg this_site = this_site_arg());
}}    // namespace hpx::collectives

// clang-format on
#else

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_distributed/async.hpp>
#include <hpx/async_local/dataflow.hpp>
#include <hpx/collectives/argument_types.hpp>
#include <hpx/collectives/create_communicator.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx::traits {

    ///////////////////////////////////////////////////////////////////////////
    // support for broadcast
    namespace communication {
        struct broadcast_tag;
    }    // namespace communication

    template <typename Communicator>
    struct communication_operation<Communicator, communication::broadcast_tag>
    {
        template <typename Result>
        static Result get(Communicator& communicator, std::size_t which,
            std::size_t generation)
        {
            using data_type = typename Result::result_type;

            return communicator.template handle_data<data_type>(
                which, generation,
                // step function (invoked for each get)
                nullptr,
                // finalizer (invoked after all sites have checked in)
                [](auto& data, bool&) {
                    return Communicator::template handle_bool<data_type>(
                        data[0]);
                },
                1);
        }

        template <typename Result, typename T>
        static Result set(Communicator& communicator, std::size_t which,
            std::size_t generation, T&& t)
        {
            return communicator.template handle_data<std::decay_t<T>>(
                which, generation,
                // step function (invoked once for set)
                [&](auto& data) { data[0] = HPX_FORWARD(T, t); },
                // finalizer (invoked after all sites have checked in)
                [](auto& data, bool&) {
                    return Communicator::template handle_bool<std::decay_t<T>>(
                        data[0]);
                },
                1);
        }
    };
}    // namespace hpx::traits

namespace hpx::collectives {

    template <typename T>
    hpx::future<std::decay_t<T>> broadcast_to(communicator fid,
        T&& local_result, this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg())
    {
        using arg_type = std::decay_t<T>;

        if (this_site == static_cast<std::size_t>(-1))
        {
            this_site = static_cast<std::size_t>(agas::get_locality_id());
        }
        if (generation == 0)
        {
            return hpx::make_exceptional_future<arg_type>(HPX_GET_EXCEPTION(
                hpx::error::bad_parameter, "hpx::collectives::broadcast_to",
                "the generation number shouldn't be zero"));
        }

        auto broadcast_data =
            [local_result = HPX_FORWARD(T, local_result), this_site,
                generation](communicator&& c) mutable -> hpx::future<arg_type> {
            using action_type =
                detail::communicator_server::communication_set_action<
                    traits::communication::broadcast_tag, hpx::future<arg_type>,
                    arg_type>;

            // explicitly unwrap returned future
            hpx::future<arg_type> result = async(action_type(), c, this_site,
                generation, HPX_MOVE(local_result));

            if (!result.is_ready())
            {
                // make sure id is kept alive as long as the returned future
                traits::detail::get_shared_state(result)->set_on_completed(
                    [client = HPX_MOVE(c)]() { HPX_UNUSED(client); });
            }

            return result;
        };

        return fid.then(hpx::launch::sync, HPX_MOVE(broadcast_data));
    }

    template <typename T>
    hpx::future<std::decay_t<T>> broadcast_to(communicator fid,
        T&& local_result, generation_arg generation,
        this_site_arg this_site = this_site_arg())
    {
        return broadcast_to(
            HPX_MOVE(fid), HPX_FORWARD(T, local_result), this_site, generation);
    }

    template <typename T>
    hpx::future<std::decay_t<T>> broadcast_to(char const* basename,
        T&& local_result, num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg())
    {
        return broadcast_to(create_communicator(basename, num_sites, this_site,
                                generation, root_site_arg(this_site.argument_)),
            HPX_FORWARD(T, local_result), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<T> broadcast_from(communicator fid,
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg())
    {
        if (this_site == static_cast<std::size_t>(-1))
        {
            this_site = static_cast<std::size_t>(agas::get_locality_id());
        }
        if (generation == 0)
        {
            return hpx::make_exceptional_future<T>(HPX_GET_EXCEPTION(
                hpx::error::bad_parameter, "hpx::collectives::broadcast_from",
                "the generation number shouldn't be zero"));
        }

        auto broadcast_data = [this_site, generation](
                                  communicator&& c) -> hpx::future<T> {
            using action_type =
                detail::communicator_server::communication_get_action<
                    traits::communication::broadcast_tag, hpx::future<T>>;

            // make sure id is kept alive as long as the returned future,
            // explicitly unwrap returned future
            hpx::future<T> result =
                async(action_type(), c, this_site, generation);

            if (!result.is_ready())
            {
                traits::detail::get_shared_state(result)->set_on_completed(
                    [client = HPX_MOVE(c)]() { HPX_UNUSED(client); });
            }

            return result;
        };

        return fid.then(hpx::launch::sync, HPX_MOVE(broadcast_data));
    }

    template <typename T>
    hpx::future<T> broadcast_from(communicator fid, generation_arg generation,
        this_site_arg this_site = this_site_arg())
    {
        return broadcast_from<T>(HPX_MOVE(fid), this_site, generation);
    }

    template <typename T>
    hpx::future<T> broadcast_from(char const* basename,
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg())
    {
        HPX_ASSERT(this_site != root_site);
        return broadcast_from<T>(create_communicator(basename, num_sites_arg(),
                                     this_site, generation, root_site),
            this_site);
    }
}    // namespace hpx::collectives

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BROADCAST_DECLARATION(...) /**/

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BROADCAST(...)             /**/

#endif    // !HPX_COMPUTE_DEVICE_CODE
#endif    // DOXYGEN
