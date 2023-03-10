//  Copyright (c) 2019-2023 Hartmut Kaiser
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
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the all_reduce operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the all_reduce operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    ///
    /// \note       The generation values from corresponding \a reduce_here and
    ///             \a reduce_there have to match.
    ///
    /// \returns    This function returns a future holding a value calculated
    ///             based on the values send by all participating sites. It will
    ///             become ready once the all_reduce operation has been completed.
    ///
    template <typename T, typename F>
    hpx::future<decay_t<T>> reduce_here(
        communicator comm, T&& local_result, F&& op,
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
    /// \note       The generation values from corresponding \a reduce_here and
    ///             \a reduce_there have to match.
    ///
    /// \returns    This function returns a future holding a value calculated
    ///             based on the values send by all participating sites. It will
    ///             become ready once the all_reduce operation has been completed.
    ///
    template <typename T, typename F>
    hpx::future<decay_t<T>> reduce_here(
        communicator comm, T&& local_result, F&& op,
        generation_arg generation,
        this_site_arg this_site = this_site_arg());

    /// Reduce a given value at the given call site
    ///
    /// This function transmits the value given by \a result to a central reduce
    /// site (where the corresponding \a reduce_here is executed)
    ///
    /// \param  basename    The base name identifying the reduction operation
    /// \param  result      A future referring to the value to transmit to the
    ///                     central reduction point from this call site.
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
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the all_reduce operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the all_reduce operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    ///
    /// \note       The generation values from corresponding \a reduce_here and
    ///             \a reduce_there have to match.
    ///
    /// \returns    This function returns a future holding a value calculated
    ///             based on the values send by all participating sites. It will
    ///             become ready once the all_reduce operation has been completed.
    ///
    template <typename T>
    hpx::future<void> reduce_there(
        communicator comm, T&& local_result,
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg());

    /// Reduce a given value at the given call site
    ///
    /// This function transmits the value given by \a result to a central reduce
    /// site (where the corresponding \a reduce_here is executed)
    ///
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  local_result A value to reduce on the central reduction point
    ///                     from this call site.
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
    /// \note       The generation values from corresponding \a reduce_here and
    ///             \a reduce_there have to match.
    ///
    /// \returns    This function returns a future holding a value calculated
    ///             based on the values send by all participating sites. It will
    ///             become ready once the all_reduce operation has been completed.
    ///
    template <typename T>
    hpx::future<void> reduce_there(
        communicator comm, T&& local_result,
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
#include <hpx/collectives/argument_types.hpp>
#include <hpx/collectives/create_communicator.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/parallel/algorithms/reduce.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx::traits {

    namespace communication {

        struct reduce_tag;
    }    // namespace communication

    ///////////////////////////////////////////////////////////////////////////
    // support for all_reduce
    template <typename Communicator>
    struct communication_operation<Communicator, communication::reduce_tag>
    {
        template <typename Result, typename T, typename F>
        static Result get(Communicator& communicator, std::size_t which,
            std::size_t generation, T&& t, F&& op)
        {
            return communicator.template handle_data<std::decay_t<T>>(
                which, generation,
                // step function (invoked once for get)
                [&](auto& data) { data[which] = HPX_FORWARD(T, t); },
                // finalizer (invoked after all data has been received)
                [op = HPX_FORWARD(F, op)](auto& data, bool&) mutable {
                    HPX_ASSERT(!data.empty());
                    if (data.size() > 1)
                    {
                        return Communicator::template handle_bool<
                            std::decay_t<T>>(hpx::reduce(++data.begin(),
                            data.end(), data[0], HPX_FORWARD(F, op)));
                    }
                    return Communicator::template handle_bool<std::decay_t<T>>(
                        data[0]);
                });
        }

        template <typename Result, typename T>
        static Result set(Communicator& communicator, std::size_t which,
            std::size_t generation, T&& t)
        {
            return communicator.template handle_data<std::decay_t<T>>(
                which, generation,
                // step function (invoked for each set)
                [&](auto& data) { data[which] = HPX_FORWARD(T, t); },
                // finalizer (invoked after all data has been received)
                nullptr);
        }
    };
}    // namespace hpx::traits

namespace hpx::collectives {

    ///////////////////////////////////////////////////////////////////////////
    // reduce plain values
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> reduce_here(communicator fid, T&& local_result,
        F&& op, this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg())
    {
        using arg_type = std::decay_t<T>;

        if (this_site == static_cast<std::size_t>(-1))
        {
            this_site = agas::get_locality_id();
        }
        if (generation == 0)
        {
            return hpx::make_exceptional_future<arg_type>(HPX_GET_EXCEPTION(
                hpx::error::bad_parameter, "hpx::collectives::reduce_here",
                "the generation number shouldn't be zero"));
        }

        auto reduction_data =
            [local_result = HPX_FORWARD(T, local_result),
                op = HPX_FORWARD(F, op), this_site,
                generation](communicator&& c) mutable -> hpx::future<arg_type> {
            using func_type = std::decay_t<F>;
            using action_type =
                detail::communicator_server::communication_get_action<
                    traits::communication::reduce_tag, hpx::future<arg_type>,
                    arg_type, func_type>;

            // explicitly unwrap returned future
            hpx::future<arg_type> result = async(action_type(), c, this_site,
                generation, HPX_FORWARD(T, local_result), HPX_FORWARD(F, op));

            if (!result.is_ready())
            {
                // make sure id is kept alive as long as the returned future
                traits::detail::get_shared_state(result)->set_on_completed(
                    [client = HPX_MOVE(c)]() { HPX_UNUSED(client); });
            }

            return result;
        };

        return fid.then(hpx::launch::sync, HPX_MOVE(reduction_data));
    }

    template <typename T, typename F>
    hpx::future<std::decay_t<T>> reduce_here(communicator fid, T&& local_result,
        F&& op, generation_arg generation,
        this_site_arg this_site = this_site_arg())
    {
        return reduce_here(HPX_MOVE(fid), HPX_FORWARD(T, local_result),
            HPX_FORWARD(F, op), this_site, generation);
    }

    template <typename T, typename F>
    hpx::future<std::decay_t<T>> reduce_here(char const* basename, T&& result,
        F&& op, num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg())
    {
        return reduce_here(create_communicator(basename, num_sites, this_site,
                               generation, root_site_arg(this_site.argument_)),
            HPX_FORWARD(T, result), HPX_FORWARD(F, op), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    // reduce plain values
    template <typename T>
    hpx::future<void> reduce_there(communicator fid, T&& local_result,
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg())
    {
        if (this_site == static_cast<std::size_t>(-1))
        {
            this_site = agas::get_locality_id();
        }
        if (generation == 0)
        {
            return hpx::make_exceptional_future<void>(HPX_GET_EXCEPTION(
                hpx::error::bad_parameter, "hpx::collectives::reduce_there",
                "the generation number shouldn't be zero"));
        }

        auto reduction_data =
            [local_result = HPX_FORWARD(T, local_result), this_site,
                generation](communicator&& c) mutable -> hpx::future<void> {
            using action_type =
                detail::communicator_server::communication_set_action<
                    traits::communication::reduce_tag, hpx::future<void>,
                    std::decay_t<T>>;

            // explicitly unwrap returned future
            hpx::future<void> result = async(action_type(), c, this_site,
                generation, HPX_MOVE(local_result));

            if (!result.is_ready())
            {
                // make sure id is kept alive as long as the returned future
                traits::detail::get_shared_state(result)->set_on_completed(
                    [client = HPX_MOVE(c)]() { HPX_UNUSED(client); });
            }

            return result;
        };

        return fid.then(hpx::launch::sync, HPX_MOVE(reduction_data));
    }

    template <typename T>
    hpx::future<void> reduce_there(communicator fid, T&& local_result,
        generation_arg generation, this_site_arg this_site = this_site_arg())
    {
        return reduce_there(
            HPX_MOVE(fid), HPX_FORWARD(T, local_result), this_site, generation);
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
            HPX_FORWARD(T, local_result), this_site);
    }
}    // namespace hpx::collectives

#endif    // !HPX_COMPUTE_DEVICE_CODE
#endif    // DOXYGEN
