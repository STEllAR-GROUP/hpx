//  Copyright (c) 2019-2025 Hartmut Kaiser
//  Copyright (c) 2026 Anshuman Agrawal
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
    /// \param  result      The value to transmit to all
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
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \param root_site    The site that is responsible for creating the
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
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  result      The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param  this_site   The sequence number of this invocation (usually
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
    hpx::future<std::decay_t<T>>
    all_reduce(communicator comm,
        T&& result, F&& op, this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg());

    /// AllReduce a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  result      The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the all_reduce operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the all_reduce operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    /// \param  this_site   The sequence number of this invocation (usually
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
        T&& result, F&& op, generation_arg generation,
        this_site_arg this_site = this_site_arg());

    /// AllReduce a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  policy      The execution policy specifying synchronous execution.
    /// \param  basename    The base name identifying the all_reduce operation
    /// \param  result      The value to transmit to all
    ///                     participating sites from this call site.
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
    /// \param root_site   The site that is responsible for creating the
    ///                     all_reduce support object. This value is optional
    ///                     and defaults to '0' (zero).
    ///
    /// \returns    This function returns a vector with all values send by all
    ///             participating sites. This function executes synchronously and
    ///             directly returns the result.
    ///
    template <typename T, typename F>
    decltype(auto) all_reduce(hpx::launch::sync_policy, char const* basename,
        T&& result, F&& op, num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg());

    /// AllReduce a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  policy      The execution policy specifying synchronous execution.
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  result      The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param  this_site   The sequence number of this invocation (usually
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
    /// \returns    This function returns a vector with all values send by all
    ///             participating sites. This function executes synchronously and
    ///             directly returns the result.
    ///
    template <typename T, typename F>
    decltype(auto) all_reduce(hpx::launch::sync_policy, communicator comm,
        T&& result, F&& op, this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg());

    /// AllReduce a set of values from different call sites
    ///
    /// This function receives a set of values from all call sites operating on
    /// the given base name.
    ///
    /// \param  policy      The execution policy specifying synchronous execution.
    /// \param  comm        A communicator object returned from \a create_communicator
    /// \param  result      The value to transmit to all
    ///                     participating sites from this call site.
    /// \param  op          Reduction operation to apply to all values supplied
    ///                     from all participating sites
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the all_reduce operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the all_reduce operation on the
    ///                     given base name has to be performed more than once.
    ///                     The generation number (if given) must be a positive
    ///                     number greater than zero.
    /// \param  this_site   The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a vector with all values send by all
    ///             participating sites. This function executes synchronously and
    ///             directly returns the result.
    ///
    template <typename T, typename F>
    decltype(auto) all_reduce(hpx::launch::sync_policy, communicator comm,
        T&& result, F&& op, generation_arg generation,
        this_site_arg this_site = this_site_arg());
}}    // namespace hpx::collectives

// clang-format on
#else

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/assert.hpp>
#include <hpx/async_distributed/async.hpp>
#include <hpx/collectives/argument_types.hpp>
#include <hpx/collectives/broadcast.hpp>
#include <hpx/collectives/create_communicator.hpp>
#include <hpx/collectives/reduce.hpp>
#include <hpx/modules/algorithms.hpp>
#include <hpx/modules/async_base.hpp>
#include <hpx/modules/components_base.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/type_support.hpp>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::traits {

    namespace communication {

        struct all_reduce_tag;

        template <>
        struct communicator_data<all_reduce_tag>
        {
            HPX_EXPORT static char const* name() noexcept;
        };
    }    // namespace communication

    ///////////////////////////////////////////////////////////////////////////
    // support for all_reduce
    template <typename Communicator>
    struct communication_operation<Communicator, communication::all_reduce_tag>
    {
        template <typename Result, typename T, typename F>
        static Result get(Communicator& communicator, std::size_t which,
            std::size_t generation, T&& t, F&& op)
        {
            return communicator.template handle_data<std::decay_t<T>>(
                communication::communicator_data<
                    communication::all_reduce_tag>::name(),
                which, generation,
                // step function (invoked for each get)
                [&t](auto& data, std::size_t which) {
                    data[which] = HPX_FORWARD(T, t);
                },
                // finalizer (invoked non-concurrently after all data has been
                // received)
                [op = HPX_FORWARD(F, op)](
                    auto& data, bool& data_available, std::size_t) mutable {
                    HPX_ASSERT(!data.empty());

                    if constexpr (!std::is_same_v<std::decay_t<T>, bool>)
                    {
                        if (!data_available && data.size() > 1)
                        {
                            // compute reduction result only once
                            auto it = data.begin();
                            data[0] = hpx::reduce(
                                ++it, data.end(), data[0], HPX_FORWARD(F, op));
                            data_available = true;
                        }
                        return data[0];
                    }
                    else
                    {
                        if (!data_available && data.size() > 1)
                        {
                            // compute reduction result only once
                            auto it = data.begin();
                            data[0] = hpx::reduce(++it, data.end(),
                                static_cast<bool>(data[0]),
                                [&](auto lhs, auto rhs) {
                                    return HPX_FORWARD(F, op)(
                                        static_cast<bool>(lhs),
                                        static_cast<bool>(rhs));
                                });
                            data_available = true;
                        }
                        return static_cast<bool>(data[0]);
                    }
                });
        }
    };
}    // namespace hpx::traits

namespace hpx::collectives {

    ////////////////////////////////////////////////////////////////////////////
    // all_reduce plain values
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> all_reduce(communicator fid, T&& local_result,
        F&& op, this_site_arg this_site = this_site_arg(),
        generation_arg const generation = generation_arg())
    {
        using arg_type = std::decay_t<T>;

        if (this_site.is_default())
        {
            this_site = agas::get_locality_id();
        }
        if (generation == 0)
        {
            return hpx::make_exceptional_future<arg_type>(HPX_GET_EXCEPTION(
                hpx::error::bad_parameter, "hpx::collectives::all_reduce",
                "the generation number shouldn't be zero"));
        }

        // Handle operation right away if there is only one value.
        if (auto [num_sites, comm_site] = fid.get_info(); num_sites == 1)
        {
            if (this_site != comm_site)
            {
                return hpx::make_exceptional_future<arg_type>(HPX_GET_EXCEPTION(
                    hpx::error::bad_parameter, "hpx::collectives::all_reduce",
                    "the local site should be zero if only one site is "
                    "involved"));
            }

            return hpx::make_ready_future(HPX_FORWARD(T, local_result));
        }

        auto all_reduce_data =
            [local_result = HPX_FORWARD(T, local_result),
                op = HPX_FORWARD(F, op), generation,
                this_site](communicator&& c) mutable -> hpx::future<arg_type> {
            using func_type = std::decay_t<F>;
            using action_type =
                detail::communicator_server::communication_get_direct_action<
                    traits::communication::all_reduce_tag,
                    hpx::future<arg_type>, arg_type, func_type>;

            // explicitly unwrap returned future
            hpx::future<arg_type> result = hpx::async(action_type(), c,
                this_site, generation, HPX_MOVE(local_result), HPX_MOVE(op));

            if (!result.is_ready())
            {
                // make sure id is kept alive as long as the returned future
                traits::detail::get_shared_state(result)->set_on_completed(
                    [client = HPX_MOVE(c)] { HPX_UNUSED(client); });
            }

            return result;
        };

        return fid.then(hpx::launch::sync, HPX_MOVE(all_reduce_data));
    }

    template <typename T, typename F>
    hpx::future<std::decay_t<T>> all_reduce(communicator fid, T&& local_result,
        F&& op, generation_arg const generation,
        this_site_arg const this_site = this_site_arg())
    {
        return all_reduce(HPX_MOVE(fid), HPX_FORWARD(T, local_result),
            HPX_FORWARD(F, op), this_site, generation);
    }

    template <typename T, typename F>
    hpx::future<std::decay_t<T>> all_reduce(char const* basename,
        T&& local_result, F&& op,
        num_sites_arg const num_sites = num_sites_arg(),
        this_site_arg const this_site = this_site_arg(),
        generation_arg const generation = generation_arg(),
        root_site_arg const root_site = root_site_arg())
    {
        return all_reduce(create_communicator(basename, num_sites, this_site,
                              generation, root_site),
            HPX_FORWARD(T, local_result), HPX_FORWARD(F, op), this_site);
    }

    ////////////////////////////////////////////////////////////////////////////
    template <typename T, typename F>
    decltype(auto) all_reduce(hpx::launch::sync_policy, communicator fid,
        T&& local_result, F&& op,
        this_site_arg const this_site = this_site_arg(),
        generation_arg const generation = generation_arg())
    {
        return all_reduce(HPX_MOVE(fid), HPX_FORWARD(T, local_result),
            HPX_FORWARD(F, op), this_site, generation)
            .get();
    }

    template <typename T, typename F>
    decltype(auto) all_reduce(hpx::launch::sync_policy, communicator fid,
        T&& local_result, F&& op, generation_arg const generation,
        this_site_arg const this_site = this_site_arg())
    {
        return all_reduce(HPX_MOVE(fid), HPX_FORWARD(T, local_result),
            HPX_FORWARD(F, op), this_site, generation)
            .get();
    }

    template <typename T, typename F>
    decltype(auto) all_reduce(hpx::launch::sync_policy, char const* basename,
        T&& local_result, F&& op,
        num_sites_arg const num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg const generation = generation_arg(),
        root_site_arg const root_site = root_site_arg())
    {
        return all_reduce(create_communicator(basename, num_sites, this_site,
                              generation, root_site),
            HPX_FORWARD(T, local_result), HPX_FORWARD(F, op), this_site)
            .get();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Helper: lift a scalar reduction op to work element-wise on vectors
    namespace detail {

        template <typename F>
        struct vector_reduce_op
        {
            F op_;

            template <typename T>
            std::vector<T> operator()(
                std::vector<T> const& lhs, std::vector<T> const& rhs) const
            {
                HPX_ASSERT(lhs.size() == rhs.size());
                std::vector<T> result;
                result.reserve(lhs.size());
                for (std::size_t i = 0; i != lhs.size(); ++i)
                {
                    result.push_back(HPX_INVOKE(op_, lhs[i], rhs[i]));
                }
                return result;
            }
        };
    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////////
    // Hierarchical all_reduce: reduce (bottom-up) + broadcast (top-down)
    // Uses 2k/2k+1 generation mapping: user generation k maps to
    // internal generation 2k (reduce phase) and 2k+1 (broadcast phase)

    // Scalar overload
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> all_reduce(
        hierarchical_communicator const& communicators, T&& local_result,
        F&& op, this_site_arg this_site = this_site_arg(),
        generation_arg const generation = generation_arg(),
        root_site_arg root_site = root_site_arg())
    {
        using arg_type = std::decay_t<T>;

        if (generation.is_default())
        {
            return hpx::make_exceptional_future<arg_type>(
                HPX_GET_EXCEPTION(hpx::error::bad_parameter,
                    "hpx::collectives::all_reduce (hierarchical)",
                    "hierarchical all_reduce requires an explicit generation "
                    "number for the 2k/2k+1 internal mapping"));
        }

        if (this_site.is_default())
        {
            this_site = agas::get_locality_id();
        }

        generation_arg const reduce_gen(2 * generation);
        generation_arg const broadcast_gen(2 * generation + 1);

        if (this_site.get() == root_site.get())
        {
            arg_type reduced = reduce_here(hpx::launch::sync, communicators,
                HPX_FORWARD(T, local_result), HPX_FORWARD(F, op), this_site,
                reduce_gen);

            return broadcast_to(
                communicators, HPX_MOVE(reduced), this_site, broadcast_gen);
        }
        else
        {
            reduce_there(hpx::launch::sync, communicators,
                HPX_FORWARD(T, local_result), HPX_FORWARD(F, op), this_site,
                reduce_gen);

            return broadcast_from<arg_type>(
                communicators, this_site, broadcast_gen);
        }
    }

    // Vector overload
    template <typename T, typename F>
    hpx::future<std::vector<T>> all_reduce(
        hierarchical_communicator const& communicators,
        std::vector<T>&& local_result, F&& op,
        this_site_arg this_site = this_site_arg(),
        generation_arg const generation = generation_arg(),
        root_site_arg root_site = root_site_arg())
    {
        if (generation.is_default())
        {
            return hpx::make_exceptional_future<std::vector<T>>(
                HPX_GET_EXCEPTION(hpx::error::bad_parameter,
                    "hpx::collectives::all_reduce (hierarchical, vector)",
                    "hierarchical all_reduce requires an explicit generation "
                    "number for the 2k/2k+1 internal mapping"));
        }

        if (this_site.is_default())
        {
            this_site = agas::get_locality_id();
        }

        generation_arg const reduce_gen(2 * generation);
        generation_arg const broadcast_gen(2 * generation + 1);

        detail::vector_reduce_op<std::decay_t<F>> vec_op{HPX_FORWARD(F, op)};

        if (this_site.get() == root_site.get())
        {
            std::vector<T> reduced = reduce_here(hpx::launch::sync, communicators,
                HPX_MOVE(local_result), HPX_MOVE(vec_op),
                this_site, reduce_gen);

            return broadcast_to(
                communicators, HPX_MOVE(reduced), this_site, broadcast_gen);
        }
        else
        {
            reduce_there(hpx::launch::sync, communicators,
                HPX_MOVE(local_result), HPX_MOVE(vec_op), this_site,
                reduce_gen);

            return broadcast_from<std::vector<T>>(
                communicators, this_site, broadcast_gen);
        }
    }

    // Sync version
    template <typename T, typename F>
    std::decay_t<T> all_reduce(hpx::launch::sync_policy,
        hierarchical_communicator const& communicators, T&& local_result,
        F&& op, this_site_arg const this_site = this_site_arg(),
        generation_arg const generation = generation_arg(),
        root_site_arg root_site = root_site_arg())
    {
        return all_reduce(communicators, HPX_FORWARD(T, local_result),
            HPX_FORWARD(F, op), this_site, generation, root_site)
            .get();
    }
}    // namespace hpx::collectives

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ALLREDUCE_DECLARATION(...) /**/

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ALLREDUCE(...)             /**/

#endif    // !HPX_COMPUTE_DEVICE_CODE
#endif    // DOXYGEN
