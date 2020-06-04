//  Copyright (c) 2014-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file scatter.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace lcos {

    /// Scatter (receive) a set of values to different call sites
    ///
    /// This function receives an element of a set of values operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the scatter operation
    /// \param  result      A future referring to the value to transmit to the
    ///                     central scatter point from this call site.
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
    /// \note       Each scatter operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_SCATTER macro to define the
    ///             necessary internal facilities used by \a scatter_from and
    ///             \a scatter_to
    ///
    /// \returns    This function returns a future holding a the
    ///             scattered value. It will become ready once the scatter
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<T> scatter_from(char const* basename,
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0);

    /// Scatter (send) a part of the value set at the given call site
    ///
    /// This function transmits the value given by \a result to a central scatter
    /// site (where the corresponding \a scatter_from is executed)
    ///
    /// \param  basename    The base name identifying the scatter operation
    /// \param  result      A future referring to the value to transmit to the
    ///                     central scatter point from this call site.
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
    /// \note       Each scatter operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_SCATTER macro to define the
    ///             necessary internal facilities used by \a scatter_from and
    ///             \a scatter_to
    ///
    /// \returns    This function returns a future holding a the
    ///             scattered value. It will become ready once the scatter
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<T> scatter_to(char const* basename,
        hpx::future<std::vector<T>> result,
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0);

    /// Scatter (receive) a set of values to different call sites
    ///
    /// This function receives an element of a set of values operating on
    /// the given base name.
    ///
    /// \param  basename    The base name identifying the scatter operation
    /// \param  result      The value to transmit to the central scatter point
    ///                     from this call site.
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
    /// \note       Each scatter operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_SCATTER macro to define the
    ///             necessary internal facilities used by \a scatter_from and
    ///             \a scatter_to
    ///
    /// \returns    This function returns a future holding a the
    ///             scattered value. It will become ready once the scatter
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<T> scatter_from(char const* basename,
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0);

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
    /// \note       Each scatter operation has to be accompanied with a unique
    ///             usage of the \a HPX_REGISTER_SCATTER macro to define the
    ///             necessary internal facilities used by \a scatter_from and
    ///             \a scatter_to
    ///
    /// \returns    This function returns a future holding a the
    ///             scattered value. It will become ready once the scatter
    ///             operation has been completed.
    ///
    template <typename T>
    hpx::future<T> scatter_to(char const* basename,
        std::vector<T> const& result, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1));

    template <typename T>
    hpx::future<T> scatter_to(char const* basename, std::vector<T>&& result,
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0);

/// \def HPX_REGISTER_SCATTER_DECLARATION(type, name)
///
/// \brief Declare a scatter object named \a name for a given data type \a type.
///
/// The macro \a HPX_REGISTER_SCATTER_DECLARATION can be used to declare
/// all facilities necessary for a (possibly remote) scatter operation.
///
/// The parameter \a type specifies for which data type the scatter
/// operations should be enabled.
///
/// The (optional) parameter \a name should be a unique C-style identifier
/// which will be internally used to identify a particular scatter operation.
/// If this defaults to \a \<type\>_scatter if not specified.
///
/// \note The macro \a HPX_REGISTER_SCATTER_DECLARATION can be used with 1
///       or 2 arguments. The second argument is optional and defaults to
///       \a \<type\>_scatter.
///
#define HPX_REGISTER_SCATTER_DECLARATION(type, name)

/// \def HPX_REGISTER_SCATTER(type, name)
///
/// \brief Define a scatter object named \a name for a given data type \a type.
///
/// The macro \a HPX_REGISTER_SCATTER can be used to define
/// all facilities necessary for a (possibly remote) scatter operation.
///
/// The parameter \a type specifies for which data type the scatter
/// operations should be enabled.
///
/// The (optional) parameter \a name should be a unique C-style identifier
/// which will be internally used to identify a particular scatter operation.
/// If this defaults to \a \<type\>_scatter if not specified.
///
/// \note The macro \a HPX_REGISTER_SCATTER can be used with 1
///       or 2 arguments. The second argument is optional and defaults to
///       \a \<type\>_scatter.
///
#define HPX_REGISTER_SCATTER(type, name)
}}    // namespace hpx::lcos
#else

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/async_distributed/dataflow.hpp>
#include <hpx/basic_execution.hpp>
#include <hpx/collectives/detail/communicator.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/acquire_shared_state.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime_local/get_num_localities.hpp>
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
    // support for scatter
    namespace communication {
        struct scatter_tag;
    }

    template <typename Communicator>
    struct communication_operation<Communicator, communication::scatter_tag>
      : std::enable_shared_from_this<
            communication_operation<Communicator, communication::scatter_tag>>
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
            auto on_ready = [this_ = std::move(this_), which](
                                shared_future<void>&& f) -> arg_type {
                f.get();    // propagate any exceptions

                auto& communicator = this_->communicator_;

                std::unique_lock<mutex_type> l(communicator.mtx_);
                return std::move(communicator.data_[which]);
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

        template <typename Result, typename T>
        Result set(std::size_t which, std::vector<T> t)
        {
            using mutex_type = typename Communicator::mutex_type;

            std::unique_lock<mutex_type> l(communicator_.mtx_);
            util::ignore_while_checking<std::unique_lock<mutex_type>> il(&l);

            communicator_.gate_.synchronize(1, l);
            communicator_.data_ = std::move(t);
            auto result = std::move(communicator_.data_[which]);

            if (communicator_.gate_.set(which, l))
            {
                // this is a one-shot object (generations counters are not
                // supported), unregister ourselves (but only once)
                hpx::unregister_with_basename(
                    std::move(communicator_.name_), communicator_.site_)
                    .get();
            }
            return result;
        }

        Communicator& communicator_;
    };
}}    // namespace hpx::traits

namespace hpx { namespace lcos {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<hpx::id_type> create_scatterer(char const* basename,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
    {
        return detail::create_communicator<T>(
            basename, num_sites, generation, this_site, num_sites);
    }

    ///////////////////////////////////////////////////////////////////////////
    // destination site needs to be handled differently
    template <typename T>
    hpx::future<T> scatter_from(hpx::future<hpx::id_type>&& fid,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        auto scatter_data =
            [this_site](hpx::future<hpx::id_type>&& fid) -> hpx::future<T> {
            using action_type = typename detail::communicator_server<T>::
                template communication_get_action<
                    traits::communication::scatter_tag, hpx::future<T>>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = fid.get();
            auto result = async(action_type(), id, this_site);

            traits::detail::get_shared_state(result)->set_on_completed(
                [id = std::move(id)]() { HPX_UNUSED(id); });

            return result;
        };

        return fid.then(hpx::launch::sync, std::move(scatter_data));
    }

    template <typename T>
    hpx::future<T> scatter_from(char const* basename,
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0)
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        std::string name(basename);
        if (generation != std::size_t(-1))
        {
            name += std::to_string(generation) + "/";
        }

        return scatter_from<T>(
            hpx::find_from_basename(std::move(name), root_site), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    // scatter_to
    template <typename T>
    hpx::future<T> scatter_to(hpx::future<hpx::id_type>&& fid,
        hpx::future<std::vector<T>>&& result,
        std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        auto scatter_to_data =
            [this_site](hpx::future<hpx::id_type>&& fid,
                hpx::future<std::vector<T>>&& local_result) -> hpx::future<T> {
            using action_type = typename detail::communicator_server<T>::
                template communication_set_action<
                    traits::communication::scatter_tag, T, std::vector<T>>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = fid.get();
            auto result =
                async(action_type(), id, this_site, local_result.get());

            traits::detail::get_shared_state(result)->set_on_completed(
                [id = std::move(id)]() { HPX_UNUSED(id); });

            return result;
        };

        return dataflow(hpx::launch::sync, std::move(scatter_to_data),
            std::move(fid), std::move(result));
    }

    template <typename T>
    hpx::future<T> scatter_to(char const* basename,
        hpx::future<std::vector<T>>&& result,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
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

        return scatter_to(
            create_scatterer<T>(basename, num_sites, generation, this_site),
            std::move(result), this_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    // scatter plain values
    template <typename T>
    hpx::future<T> scatter_to(hpx::future<hpx::id_type>&& fid,
        std::vector<T>&& local_result, std::size_t this_site = std::size_t(-1))
    {
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }

        auto scatter_to_data_direct =
            [this_site](hpx::future<hpx::id_type>&& fid,
                std::vector<T>&& local_result) -> hpx::future<T> {
            using action_type = typename detail::communicator_server<T>::
                template communication_set_action<
                    traits::communication::scatter_tag, T, std::vector<T>>;

            // make sure id is kept alive as long as the returned future
            hpx::id_type id = fid.get();
            auto result =
                async(action_type(), id, this_site, std::move(local_result));

            traits::detail::get_shared_state(result)->set_on_completed(
                [id = std::move(id)]() { HPX_UNUSED(id); });

            return result;
        };

        return dataflow(std::move(scatter_to_data_direct), std::move(fid),
            std::move(local_result));
    }

    template <typename T>
    hpx::future<T> scatter_to(char const* basename,
        std::vector<T>&& local_result, std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1))
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

        std::string name(basename);
        if (generation != std::size_t(-1))
        {
            name += std::to_string(generation) + "/";
        }

        return scatter_to(
            create_scatterer<T>(basename, num_sites, generation, this_site),
            std::move(local_result), this_site);
    }
}}    // namespace hpx::lcos

////////////////////////////////////////////////////////////////////////////////
namespace hpx {
    using lcos::scatter_from;
    using lcos::scatter_to;
}    // namespace hpx

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_SCATTER_DECLARATION(...) /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_SCATTER(...)                                              \
    HPX_REGISTER_SCATTER_(__VA_ARGS__)                                         \
    /**/

#define HPX_REGISTER_SCATTER_(...)                                             \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                  \
        HPX_REGISTER_SCATTER_, HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))        \
    /**/

#define HPX_REGISTER_SCATTER_1(type)                                           \
    HPX_REGISTER_SCATTER_2(type, HPX_PP_CAT(type, _scatter))                   \
    /**/

#define HPX_REGISTER_SCATTER_2(type, name)                                     \
    typedef hpx::components::component<                                        \
        hpx::lcos::detail::communicator_server<type>>                          \
        HPX_PP_CAT(scatter_, name);                                            \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(scatter_, name))                         \
    /**/

#endif    // COMPUTE_HOST_CODE
#endif    // DOXYGEN
