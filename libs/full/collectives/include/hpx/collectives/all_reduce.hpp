//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file all_reduce.hpp

#pragma once

#if defined(DOXYGEN)
// clang-format off
namespace hpx { namespace lcos {

    /// AllReduce a set of values from different call sites
    ///
    /// This function receives a set of values that are the result of applying
    /// a given operator on values supplied from all call sites operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the all_reduce operation
    /// \param  local_result A future referring to the value to transmit to all
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
    /// \note       Each all_reduce operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_ALLREDUCE macro to define the
    ///             necessary internal facilities used by \a all_reduce.
    ///
    /// \returns    This function returns a future holding a value calculated
    ///             based on the values send by all participating sites. It will
    ///             become ready once the all_reduce operation has been completed.
    ///
    template <typename T, typename F>
    hpx::future<T> all_reduce(char const* basename, hpx::future<T> result,
        F&& op, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0);

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
    /// \note       Each all_reduce operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_ALLREDUCE macro to define the
    ///             necessary internal facilities used by \a all_reduce.
    ///
    /// \returns    This function returns a future holding a vector with all
    ///             values send by all participating sites. It will become
    ///             ready once the all_reduce operation has been completed.
    ///
    template <typename T, typename F>
    hpx::future<std::decay_t<T>> all_reduce(char const* basename, T&& result,
        F&& op, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0);
}}    // namespace hpx::lcos

// clang-format on
#else

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_local/dataflow.hpp>
#include <hpx/collectives/detail/communicator.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/acquire_shared_state.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/parallel/algorithms/reduce.hpp>
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime_local/get_num_localities.hpp>
#include <hpx/thread_support/assert_owns_lock.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
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
        communication_operation(Communicator& comm)
          : communicator_(comm)
        {
        }

        template <typename Result, typename T, typename F>
        Result get(std::size_t which, T&& t, F&& op)
        {
            using arg_type = typename std::decay<T>::type;
            using mutex_type = typename Communicator::mutex_type;
            using lock_type = std::unique_lock<mutex_type>;

            auto this_ = this->shared_from_this();
            auto on_ready =
                [this_ = std::move(this_), op = std::forward<F>(op)](
                    hpx::shared_future<void> f) mutable -> arg_type {
                f.get();    // propagate any exceptions

                auto& communicator = this_->communicator_;

                lock_type l(communicator.mtx_);
                util::ignore_while_checking<lock_type> il(&l);

                auto& data = communicator.template access_data<arg_type>(l);

                auto it = data.begin();
                return hpx::reduce(
                    hpx::execution::par, ++it, data.end(), *data.begin(), op);
            };

            lock_type l(communicator_.mtx_);
            util::ignore_while_checking<lock_type> il(&l);

            hpx::future<arg_type> f =
                communicator_.gate_.get_shared_future(l).then(
                    hpx::launch::sync, std::move(on_ready));

            communicator_.gate_.synchronize(1, l);

            auto& data = communicator_.template access_data<arg_type>(l);
            data[which] = std::forward<T>(t);

            if (communicator_.gate_.set(which, l))
            {
                HPX_ASSERT_DOESNT_OWN_LOCK(l);
                {
                    lock_type l(communicator_.mtx_);
                    communicator_.invalidate_data(l);
                }

                // this is a one-shot object (generations counters are not
                // supported), unregister ourselves (but only once)
                hpx::unregister_with_basename(
                    std::move(communicator_.name_), communicator_.site_)
                    .get();
            }
            return f;
        }

        Communicator& communicator_;
    };
}}    // namespace hpx::traits

namespace hpx { namespace lcos {

    ///////////////////////////////////////////////////////////////////////////
    inline hpx::future<hpx::id_type> create_all_reduce(char const* basename,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
    {
        return detail::create_communicator(
            basename, num_sites, generation, this_site);
    }

    ////////////////////////////////////////////////////////////////////////////
    // destination site needs to be handled differently
    template <typename T, typename F>
    hpx::future<T> all_reduce(hpx::future<hpx::id_type>&& fid,
        hpx::future<T>&& local_result, F&& op,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        auto all_reduce_data =
            [op = std::forward<F>(op), this_site](hpx::future<hpx::id_type>&& f,
                hpx::future<T>&& local_result) mutable -> hpx::future<T> {
            using func_type = typename std::decay<F>::type;
            using action_type = typename detail::communicator_server::
                template communication_get_action<
                    traits::communication::all_reduce_tag, hpx::future<T>, T,
                    func_type>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = f.get();
            hpx::future<T> result = async(action_type(), id, this_site,
                local_result.get(), std::forward<F>(op));

            traits::detail::get_shared_state(result)->set_on_completed(
                [id = std::move(id)]() { HPX_UNUSED(id); });

            return result;
        };

        return dataflow(hpx::launch::sync, std::move(all_reduce_data),
            std::move(fid), std::move(local_result));
    }

    template <typename T, typename F>
    hpx::future<T> all_reduce(char const* basename,
        hpx::future<T>&& local_result, F&& op,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)
    {
        if (num_sites == std::size_t(-1))
        {
            num_sites = static_cast<std::size_t>(
                hpx::get_num_localities(hpx::launch::sync));
        }
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        if (this_site == root_site)
        {
            return all_reduce(
                create_all_reduce(basename, num_sites, generation, root_site),
                std::move(local_result), std::forward<F>(op), this_site);
        }

        std::string name(basename);
        if (generation != std::size_t(-1))
        {
            name += std::to_string(generation) + "/";
        }

        return all_reduce(hpx::find_from_basename(std::move(name), root_site),
            std::move(local_result), std::forward<F>(op), this_site);
    }

    ////////////////////////////////////////////////////////////////////////////
    // all_reduce plain values
    template <typename T, typename F>
    hpx::future<typename std::decay<T>::type> all_reduce(
        hpx::future<hpx::id_type>&& fid, T&& local_result, F&& op,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        using arg_type = typename std::decay<T>::type;

        auto all_reduce_data_direct =
            [op = std::forward<F>(op),
                local_result = std::forward<T>(local_result),
                this_site](hpx::future<hpx::id_type>&& f) mutable
            -> hpx::future<arg_type> {
            using func_type = typename std::decay<F>::type;
            using action_type = typename detail::communicator_server::
                template communication_get_action<
                    traits::communication::all_reduce_tag,
                    hpx::future<arg_type>, arg_type, func_type>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = f.get();
            hpx::future<arg_type> result = async(action_type(), id, this_site,
                std::forward<T>(local_result), std::forward<F>(op));

            traits::detail::get_shared_state(result)->set_on_completed(
                [id = std::move(id)]() { HPX_UNUSED(id); });

            return result;
        };

        return fid.then(hpx::launch::sync, std::move(all_reduce_data_direct));
    }

    template <typename T, typename F>
    hpx::future<typename std::decay<T>::type> all_reduce(char const* basename,
        T&& local_result, F&& op, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)
    {
        if (num_sites == std::size_t(-1))
        {
            num_sites = static_cast<std::size_t>(
                hpx::get_num_localities(hpx::launch::sync));
        }
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        if (this_site == root_site)
        {
            return all_reduce(
                create_all_reduce(basename, num_sites, generation, root_site),
                std::forward<T>(local_result), std::forward<F>(op), this_site);
        }

        std::string name(basename);
        if (generation != std::size_t(-1))
            name += std::to_string(generation) + "/";

        return all_reduce(hpx::find_from_basename(std::move(name), root_site),
            std::forward<T>(local_result), std::forward<F>(op), this_site);
    }
}}    // namespace hpx::lcos

////////////////////////////////////////////////////////////////////////////////
namespace hpx {
    using lcos::all_reduce;
    using lcos::create_all_reduce;
}    // namespace hpx

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ALLREDUCE_DECLARATION(...) /**/

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ALLREDUCE(...)             /**/

#endif    // COMPUTE_HOST_CODE
#endif    // DOXYGEN
