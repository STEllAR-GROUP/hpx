//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file broadcast.hpp

#pragma once

#if defined(DOXYGEN)
// clang-format off
namespace hpx { namespace lcos {

    /// Broadcast a value to different call sites
    ///
    /// This function sends a set of values to all call sites operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the broadcast operation
    /// \param  local_result A future referring to the value to transmit to all
    ///                     participating sites from this call site.
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the broadcast operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the broadcast operation on the
    ///                     given base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \params root_site   The site that is responsible for creating the
    ///                     broadcast support object. This value is optional
    ///                     and defaults to '0' (zero).
    ///
    /// \note       Each broadcast operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_BROADCAST macro to define the
    ///             necessary internal facilities used by \a broadcast.
    ///
    /// \returns    This function returns a future that will become
    ///             ready once the broadcast operation has been completed.
    ///
    template <typename T>
    hpx::future<void> broadcast_to(char const* basename,
        hpx::future<T>&& local_result, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)

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
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \params root_site   The site that is responsible for creating the
    ///                     broadcast support object. This value is optional
    ///                     and defaults to '0' (zero).
    ///
    /// \note       Each broadcast operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_BROADCAST macro to define the
    ///             necessary internal facilities used by \a broadcast.
    ///
    /// \returns    This function returns a future that will become
    ///             ready once the broadcast operation has been completed.
    ///
    template <typename T>
    hpx::future<void> broadcast_to(char const* basename,
        T&& local_result, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)

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
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \params root_site   The site that is responsible for creating the
    ///                     broadcast support object. This value is optional
    ///                     and defaults to '0' (zero).
    ///
    /// \note       Each broadcast operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_BROADCAST macro to define the
    ///             necessary internal facilities used by \a broadcast.
    ///
    /// \returns    This function returns a future holding the value that was
    ///             sent to all participating sites. It will become
    ///             ready once the broadcast operation has been completed.
    ///
    template <typename T>
    hpx::future<T> broadcast_from(char const* basename,
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)

/// \def HPX_REGISTER_BROADCAST_DECLARATION(type, name)
///
/// \brief Declare a broadcast object named \a name for a given data type \a type.
///
/// The macro \a HPX_REGISTER_BROADCAST_DECLARATION can be used to declare
/// all facilities necessary for a (possibly remote) broadcast operation.
///
/// The parameter \a type specifies for which data type the broadcast
/// operations should be enabled.
///
/// The (optional) parameter \a name should be a unique C-style identifier
/// that will be internally used to identify a particular broadcast operation.
/// If this defaults to \a \<type\>_broadcast if not specified.
///
/// \note The macro \a HPX_REGISTER_BROADCAST_DECLARATION can be used with 1
///       or 2 arguments. The second argument is optional and defaults to
///       \a \<type\>_broadcast.
///
#define HPX_REGISTER_BROADCAST_DECLARATION(type, name)

/// \def HPX_REGISTER_BROADCAST(type, name)
///
/// \brief Define a broadcast object named \a name for a given data type \a type.
///
/// The macro \a HPX_REGISTER_BROADCAST can be used to define
/// all facilities necessary for a (possibly remote) broadcast operation.
///
/// The parameter \a type specifies for which data type the broadcast
/// operations should be enabled.
///
/// The (optional) parameter \a name should be a unique C-style identifier
/// that will be internally used to identify a particular broadcast operation.
/// If this defaults to \a \<type\>_broadcast if not specified.
///
/// \note The macro \a HPX_REGISTER_BROADCAST can be used with 1
///       or 2 arguments. The second argument is optional and defaults to
///       \a \<type\>_broadcast.
///
#define HPX_REGISTER_BROADCAST(type, name)
}}    // namespace hpx::lcos
// clang-format on
#else

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/async/dataflow.hpp>
#include <hpx/basic_execution.hpp>
#include <hpx/collectives/detail/communicator.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/get_num_localities.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/futures/traits/acquire_shared_state.hpp>
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

    ///////////////////////////////////////////////////////////////////////////
    // support for broadcast
    namespace communication {
        struct broadcast_tag;
    }

    template <typename Communicator>
    struct communication_operation<Communicator, communication::broadcast_tag>
      : std::enable_shared_from_this<
            communication_operation<Communicator, communication::broadcast_tag>>
    {
        communication_operation(Communicator& comm)
          : communicator_(comm)
        {
        }

        template <typename Result>
        Result get(std::size_t which)
        {
            using arg_type = typename Communicator::arg_type;
            using mutex_type = typename Communicator::mutex_type;

            auto this_ = this->shared_from_this();
            auto on_ready = [this_ = std::move(this_)](
                                shared_future<void>&& f) -> arg_type {
                f.get();    // propagate any exceptions

                auto& communicator = this_->communicator_;

                arg_type data;
                {
                    std::unique_lock<mutex_type> l(communicator.mtx_);
                    data = communicator.data_[0];
                }
                return data;
            };

            std::unique_lock<mutex_type> l(communicator_.mtx_);
            util::ignore_while_checking<std::unique_lock<mutex_type>> il(&l);

            hpx::future<arg_type> f =
                communicator_.gate_.get_shared_future(l).then(
                    hpx::launch::sync, std::move(on_ready));

            communicator_.gate_.synchronize(1, l);
            if (communicator_.gate_.set(which, l))
            {
                // this is a one-shot object (generations counters are not
                // supported), unregister ourselves (but only once)
                hpx::unregister_with_basename(
                    std::move(communicator_.name_), communicator_.site_)
                    .get();
            }
            return f;
        }

        template <typename T>
        void set(std::size_t which, T&& t)
        {
            using mutex_type = typename Communicator::mutex_type;

            std::unique_lock<mutex_type> l(communicator_.mtx_);
            util::ignore_while_checking<std::unique_lock<mutex_type>> il(&l);

            communicator_.gate_.synchronize(1, l);
            communicator_.data_[0] = std::forward<T>(t);
            if (communicator_.gate_.set(which, l))
            {
                // this is a one-shot object (generations counters are not
                // supported), unregister ourselves (but only once)
                hpx::unregister_with_basename(
                    std::move(communicator_.name_), communicator_.site_)
                    .get();
            }
        }

        Communicator& communicator_;
    };
}}    // namespace hpx::traits

namespace hpx { namespace lcos {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<hpx::id_type> create_broadcast(char const* basename,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
    {
        // everybody waits for exactly one value
        return detail::create_communicator<T>(
            basename, num_sites, generation, this_site, 1);
    }

    ///////////////////////////////////////////////////////////////////////////
    // destination site needs to be handled differently
    template <typename T>
    hpx::future<void> broadcast_to(hpx::future<hpx::id_type>&& fid,
        hpx::future<T>&& local_result, std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        auto broadcast_data =
            [this_site](hpx::future<hpx::id_type>&& f,
                hpx::future<T>&& local_result) -> hpx::future<void> {
            using action_type = typename detail::communicator_server<T>::
                template communication_set_action<
                    traits::communication::broadcast_tag, T>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = f.get();
            auto result =
                async(action_type(), id, this_site, local_result.get());

            traits::detail::get_shared_state(result)->set_on_completed(
                [id = std::move(id)]() { HPX_UNUSED(id); });

            return result;
        };

        return dataflow(hpx::launch::sync, std::move(broadcast_data),
            std::move(fid), std::move(local_result));
    }

    template <typename T>
    hpx::future<void> broadcast_to(hpx::future<hpx::id_type>&& fid,
        T&& local_result, std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        using arg_type = typename std::decay<T>::type;

        auto broadcast_data =
            [this_site](hpx::future<hpx::id_type>&& f,
                arg_type&& local_result) -> hpx::future<void> {
            using action_type = typename detail::communicator_server<T>::
                template communication_set_action<
                    traits::communication::broadcast_tag, arg_type>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = f.get();
            auto result =
                async(action_type(), id, this_site, std::move(local_result));

            traits::detail::get_shared_state(result)->set_on_completed(
                [id = std::move(id)]() { HPX_UNUSED(id); });

            return result;
        };

        return dataflow(hpx::launch::sync, std::move(broadcast_data),
            std::move(fid), std::forward<T>(local_result));
    }

    template <typename T>
    hpx::future<void> broadcast_to(char const* basename,
        hpx::future<T>&& local_result, std::size_t num_sites = std::size_t(-1),
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
            return broadcast_to(create_broadcast<typename std::decay<T>::type>(
                                    basename, num_sites, generation, root_site),
                std::move(local_result), this_site);
        }

        std::string name(basename);
        if (generation != std::size_t(-1))
            name += std::to_string(generation) + "/";

        return broadcast_to(hpx::find_from_basename(std::move(name), root_site),
            std::move(local_result), this_site);
    }

    template <typename T>
    hpx::future<void> broadcast_to(char const* basename, T&& local_result,
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

        return broadcast_to(create_broadcast<typename std::decay<T>::type>(
                                basename, num_sites, generation, root_site),
            std::forward<T>(local_result), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<T> broadcast_from(hpx::future<hpx::id_type>&& fid,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        using arg_type = typename util::decay<T>::type;

        auto broadcast_data_direct =
            [this_site](hpx::future<hpx::id_type>&& fid) -> hpx::future<T> {
            using action_type = typename detail::communicator_server<T>::
                template communication_get_action<
                    traits::communication::broadcast_tag, hpx::future<T>>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = fid.get();
            auto result = async(action_type(), id, this_site);

            traits::detail::get_shared_state(result)->set_on_completed(
                [id = std::move(id)]() { HPX_UNUSED(id); });

            return result;
        };

        return dataflow(hpx::launch::sync, std::move(broadcast_data_direct),
            std::move(fid));
    }

    template <typename T>
    hpx::future<T> broadcast_from(char const* basename,
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        std::string name(basename);
        if (generation != std::size_t(-1))
            name += std::to_string(generation) + "/";

        return broadcast_from<T>(
            hpx::find_from_basename(std::move(name), root_site), this_site);
    }
}}    // namespace hpx::lcos

////////////////////////////////////////////////////////////////////////////////
namespace hpx {
    using lcos::broadcast_from;
    using lcos::broadcast_to;
    using lcos::create_broadcast;
}    // namespace hpx

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BROADCAST_DECLARATION(...) /**/

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BROADCAST(...)                                            \
    HPX_REGISTER_BROADCAST_(__VA_ARGS__)                                       \
    /**/

#define HPX_REGISTER_BROADCAST_(...)                                           \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                  \
        HPX_REGISTER_BROADCAST_, HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))      \
    /**/

#define HPX_REGISTER_BROADCAST_1(type)                                         \
    HPX_REGISTER_BROADCAST_2(type, HPX_PP_CAT(type, _broadcast))               \
    /**/

#define HPX_REGISTER_BROADCAST_2(type, name)                                   \
    typedef hpx::components::component<                                        \
        hpx::lcos::detail::communicator_server<type>>                          \
        HPX_PP_CAT(broadcast_, name);                                          \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(broadcast_, name))                       \
    /**/

#endif    // COMPUTE_HOST_CODE
#endif    // DOXYGEN
