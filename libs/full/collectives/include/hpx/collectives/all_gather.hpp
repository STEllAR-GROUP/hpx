//  Copyright (c) 2019-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file all_gather.hpp

#pragma once

#if defined(DOXYGEN)
// clang-format off
namespace hpx { namespace collectives {

    /// AllGather a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the all_gather operation
    /// \param  local_result The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the all_gather operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the all_gather operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    /// \params root_site   The site that is responsible for creating the
    ///                     all_gather support object. This value is optional
    ///                     and defaults to '0' (zero).
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             values send by all participating sites. It will become
    ///             ready once the all_gather operation has been completed.
    ///
    template <typename T>
    hpx::future<std::vector<std::decay_t<T>>>
    all_gather(char const* basename, T&& result,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg());

    /// AllGather a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  local_result The value to transmit to all
    ///                     participating sites from this call site.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the all_reduce operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the all_reduce operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             values send by all participating sites. It will become
    ///             ready once the all_gather operation has been completed.
    ///
    template <typename T>
    hpx::future<std::vector<std::decay_t<T>>>
    all_gather(communicator comm, T&& result,
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg());

    /// AllGather a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  local_result The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the all_reduce operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the all_reduce operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             values send by all participating sites. It will become
    ///             ready once the all_gather operation has been completed.
    ///
    template <typename T>
    hpx::future<std::vector<std::decay_t<T>>>
    all_gather(communicator comm, T&& result,
        generation_arg generation,
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
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::traits {

    namespace communication {
        struct all_gather_tag;
    }    // namespace communication

    ///////////////////////////////////////////////////////////////////////////
    // support for all_gather
    template <typename Communicator>
    struct communication_operation<Communicator, communication::all_gather_tag>
    {
        template <typename Result, typename T>
        static Result get(Communicator& communicator, std::size_t which,
            std::size_t generation, T&& t)
        {
            return communicator.template handle_data<std::decay_t<T>>(
                which, generation,
                // step function (invoked for each get)
                [&](auto& data) { data[which] = HPX_FORWARD(T, t); },
                // finalizer (invoked after all data has been received)
                [](auto& data, bool&) { return data; });
        }
    };
}    // namespace hpx::traits

namespace hpx::collectives {

    ///////////////////////////////////////////////////////////////////////////
    // all_gather plain values
    template <typename T>
    hpx::future<std::vector<std::decay_t<T>>> all_gather(communicator fid,
        T&& local_result, this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg())
    {
        using arg_type = std::decay_t<T>;

        if (this_site == static_cast<std::size_t>(-1))
        {
            this_site = agas::get_locality_id();
        }
        if (generation == 0)
        {
            return hpx::make_exceptional_future<std::vector<arg_type>>(
                HPX_GET_EXCEPTION(hpx::error::bad_parameter,
                    "hpx::collectives::all_gather",
                    "the generation number shouldn't be zero"));
        }

        auto all_gather_data = [local_result = HPX_FORWARD(T, local_result),
                                   this_site,
                                   generation](communicator&& c) mutable
            -> hpx::future<std::vector<arg_type>> {
            using action_type =
                detail::communicator_server::communication_get_action<
                    traits::communication::all_gather_tag,
                    hpx::future<std::vector<arg_type>>, arg_type>;

            // explicitly unwrap returned future
            hpx::future<std::vector<arg_type>> result = async(action_type(), c,
                this_site, generation, HPX_MOVE(local_result));

            if (!result.is_ready())
            {
                // make sure id is kept alive as long as the returned future
                traits::detail::get_shared_state(result)->set_on_completed(
                    [client = HPX_MOVE(c)]() { HPX_UNUSED(client); });
            }

            return result;
        };

        return fid.then(hpx::launch::sync, HPX_MOVE(all_gather_data));
    }

    template <typename T>
    hpx::future<std::vector<std::decay_t<T>>> all_gather(communicator fid,
        T&& local_result, generation_arg generation,
        this_site_arg this_site = this_site_arg())
    {
        return all_gather(
            HPX_MOVE(fid), HPX_FORWARD(T, local_result), this_site, generation);
    }

    template <typename T>
    hpx::future<std::vector<std::decay_t<T>>> all_gather(char const* basename,
        T&& local_result, num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg())
    {
        return all_gather(create_communicator(basename, num_sites, this_site,
                              generation, root_site),
            HPX_FORWARD(T, local_result), this_site);
    }
}    // namespace hpx::collectives

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ALLGATHER_DECLARATION(...) /**/

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ALLGATHER(...)             /**/

#endif    // !HPX_COMPUTE_DEVICE_CODE
#endif    // DOXYGEN
