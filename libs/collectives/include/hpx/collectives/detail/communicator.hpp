//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COLLECTIVES_DETAIL_COMMUNICATOR_MAY_05_2020_0232AM)
#define HPX_COLLECTIVES_DETAIL_COMMUNICATOR_MAY_05_2020_0232AM

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/assertion.hpp>
#include <hpx/basic_execution/register_locks.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/local_lcos/and_gate.hpp>
#include <hpx/parallel/algorithms/reduce.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/components/new.hpp>
#include <hpx/runtime/components/server/component_base.hpp>
#include <hpx/runtime/get_num_localities.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/type_support/decay.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace traits {

    // This type can be specialized for a particular collective operation
    template <typename Communicator, typename Operation>
    struct communication_operation;

}}    // namespace hpx::traits

namespace hpx { namespace lcos { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class communicator_server;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class communicator_server
      : public hpx::components::component_base<communicator_server<T>>
    {
        using mutex_type = lcos::local::spinlock;
        using arg_type = T;

    public:
        communicator_server()    //-V730
        {
            HPX_ASSERT(false);    // shouldn't ever be called
        }

        communicator_server(std::size_t num_sites, std::string const& name,
            std::size_t site, std::size_t num_values)
          : data_(num_values)
          , gate_(num_sites)
          , name_(name)
          , num_sites_(num_sites)
          , site_(site)
        {
            HPX_ASSERT(num_values != 0);
            HPX_ASSERT(num_sites != 0);
        }

        ///////////////////////////////////////////////////////////////////////
        // generic get action, dispatches to proper operation
        template <typename Operation, typename Result, typename... Args>
        Result get_result(std::size_t which, Args... args)
        {
            return std::make_shared<traits::communication_operation<
                communicator_server, Operation>>(*this)
                ->template get<Result>(which, std::move(args)...);
        }

        template <typename Operation, typename Result, typename... Args>
        struct communication_get_action
          : hpx::actions::make_action<Result (communicator_server::*)(
                                          std::size_t, Args...),
                &communicator_server::template get_result<Operation, Result,
                    Args...>,
                communication_get_action<Operation, Result, Args...>>::type
        {
        };

        template <typename Operation, typename... Args>
        void set_result(std::size_t which, Args... args)
        {
            return std::make_shared<traits::communication_operation<
                communicator_server, Operation>>(*this)
                ->set(which, std::move(args)...);
        }

        template <typename Operation, typename... Args>
        struct communication_set_action
          : hpx::actions::make_action<void (communicator_server::*)(
                                          std::size_t, Args...),
                &communicator_server::template set_result<Operation, Args...>,
                communication_set_action<Operation, Args...>>::type
        {
        };

    private:
        template <typename Communicator, typename Operation>
        friend struct hpx::traits::communication_operation;

    private:
        mutex_type mtx_;
        std::vector<T> data_;
        lcos::local::and_gate gate_;
        std::string name_;
        std::size_t const num_sites_;
        std::size_t const site_;
    };    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    inline hpx::future<hpx::id_type> register_communicator_name(
        hpx::future<hpx::id_type>&& f, std::string basename, std::size_t site)
    {
        hpx::id_type target = f.get();

        // Register unmanaged id to avoid cyclic dependencies, unregister
        // is done after all data has been collected in the component above.
        hpx::future<bool> result =
            hpx::register_with_basename(basename, target, site);

        return result.then(hpx::launch::sync,
            [target = std::move(target), basename = std::move(basename)](
                hpx::future<bool>&& f) -> hpx::id_type {
                bool result = f.get();
                if (!result)
                {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "hpx::lcos::detail::register_communicator_name",
                        "the given base name for the communicator "
                        "operation was already registered: " +
                            basename);
                }
                return target;
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<hpx::id_type> create_communicator(char const* basename,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1),
        std::size_t num_values = std::size_t(-1))
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
        if (num_values == std::size_t(-1))
        {
            num_values = num_sites;
        }

        std::string name(basename);
        if (generation != std::size_t(-1))
        {
            name += std::to_string(generation) + "/";
        }

        // create a new communicator_server
        using result_type = typename util::decay<T>::type;
        hpx::future<hpx::id_type> id =
            hpx::new_<detail::communicator_server<result_type>>(
                hpx::find_here(), num_sites, name, this_site, num_values);

        // register the communicator's id using the given basename
        return id.then(hpx::launch::sync,
            util::bind_back(&detail::register_communicator_name,
                std::move(name), this_site));
    }
}}}    // namespace hpx::lcos::detail

#endif    // COMPUTE_HOST_CODE
#endif
