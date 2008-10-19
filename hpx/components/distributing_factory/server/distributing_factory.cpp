//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include <hpx/components/distributing_factory/server/distributing_factory.hpp>

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the distributing_factory action
typedef hpx::lcos::base_lco_with_value<
        hpx::components::server::detail::distributing_factory::result_type 
    > create_result_type;

HPX_REGISTER_ACTION(create_result_type::set_result_action);
HPX_DEFINE_GET_COMPONENT_TYPE(create_result_type);

HPX_REGISTER_ACTION(hpx::components::server::detail::distributing_factory::create_components_action);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::server::detail::distributing_factory);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    component_type distributing_factory::value = component_invalid;

    struct lazy_result
    {
        lazy_result(naming::id_type const& prefix, 
                lcos::future_value<naming::id_type> gids,
                std::size_t count)
          : prefix_(prefix), gids_(gids), count_(count)
        {}

        naming::id_type prefix_;
        lcos::future_value<naming::id_type> gids_;
        std::size_t count_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // create a new instance of a component
    threads::thread_state distributing_factory::create_components(
        threads::thread_self& self, applier::applier& appl,
        result_type* gids, components::component_type type, 
        std::size_t count)
    {
        // get list of locality prefixes
        std::vector<naming::id_type> prefixes;
        appl.get_dgas_client().get_prefixes(prefixes);

        std::size_t created_count = 0;
        std::size_t count_on_locality = count / prefixes.size();
        if (0 == count_on_locality)
            count_on_locality = 1;

        // distribute the number of components to create evenly on all 
        // available localities
        typedef std::vector<lazy_result> future_values_type;

        // start an asynchronous operation for each of the localities
        future_values_type v;
        components::stubs::runtime_support rts(appl);

        std::vector<naming::id_type>::iterator end = prefixes.end();
        for (std::vector<naming::id_type>::iterator it = prefixes.begin(); 
             it != end; ++it)
        {
            std::size_t create = count_on_locality;
            if (created_count + create > count)
                create = count - created_count;

            lcos::future_value<naming::id_type> f (
                rts.create_component_async(*it, type, create));
            v.push_back(future_values_type::value_type(*it, f, create));

            created_count += create;
            if (created_count >= count)
                break;
        }

        // now wait for the results
        future_values_type::iterator vend = v.end();
        for (future_values_type::iterator vit = v.begin(); vit != vend; ++vit)
        {
            gids->push_back(result_type::value_type(
                (*vit).prefix_, (*vit).gids_.get_result(self), (*vit).count_));
        }

        return threads::terminated;
    }

}}}}

