//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/server/memory_block.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/assert.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE();
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::server::memory_block, memory_block);

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the memory_block actions
HPX_REGISTER_ACTION(hpx::components::server::detail::memory_block::get_action);
HPX_REGISTER_ACTION(hpx::components::server::detail::memory_block::checkout_action);
HPX_REGISTER_ACTION(hpx::components::server::detail::memory_block::checkin_action);
HPX_REGISTER_ACTION(hpx::components::server::detail::memory_block::clone_action);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::server::detail::memory_block_header);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::server::detail::memory_block);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::server::memory_block);

typedef hpx::components::memory_block_data memory_data_type;
HPX_REGISTER_ACTION(hpx::lcos::base_lco_with_value<memory_data_type>::set_result_action);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<memory_data_type>);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server { namespace detail
{
    /// Get the current data for reading
    components::memory_block_data memory_block::get() 
    {
        return components::memory_block_data(wrapper_->component_);
    }

    /// Get the current data for reading
    components::memory_block_data memory_block::checkout() 
    {
        return components::memory_block_data(wrapper_->component_);
    }

    /// Write back data
    void memory_block::checkin(components::memory_block_data const& data) 
    {
    }

    /// Clone this memory_block
    naming::id_type create_memory_block (detail::memory_block_header const* rhs)
    {
        server::memory_block* c = server::memory_block::create(rhs);
        naming::id_type gid = c->get_gid();
        if (gid) 
            return gid;

        delete c;
        HPX_THROW_EXCEPTION(hpx::duplicate_component_address,
            "global id is already bound to a different "
            "component instance");
        return naming::invalid_id;
    }

    naming::id_type memory_block::clone() 
    {
        return create_memory_block(wrapper_->component_.get());
    }

}}}}
