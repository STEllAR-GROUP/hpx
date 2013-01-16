//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/server/memory_block.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/manage_object_action.hpp>
#include <hpx/runtime/get_lva.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/assert.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE()

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the memory_block actions
HPX_REGISTER_ACTION(
    hpx::components::server::detail::memory_block::get_action,
    memory_block_get_action)
HPX_REGISTER_ACTION(
    hpx::components::server::detail::memory_block::get_config_action,
    memory_block_get_action)
HPX_REGISTER_ACTION(
    hpx::components::server::detail::memory_block::checkout_action,
    memory_block_checkout_action)
HPX_REGISTER_ACTION(
    hpx::components::server::detail::memory_block::checkin_action,
    memory_block_checkin_action)
HPX_REGISTER_ACTION(
    hpx::components::server::detail::memory_block::clone_action,
    memory_block_clone_action)

HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::components::server::detail::memory_block_header,
    hpx::components::component_memory_block)
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::components::server::detail::memory_block,
    hpx::components::component_memory_block)
HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(
    hpx::components::server::memory_block,
    hpx::components::component_memory_block)

typedef hpx::components::memory_block_data memory_data_type;
HPX_REGISTER_BASE_LCO_WITH_VALUE(
    hpx::components::memory_block_data,
    memory_data_type)

HPX_REGISTER_MANAGE_OBJECT_ACTION(
    hpx::actions::manage_object_action<boost::uint8_t>,
    manage_object_action_uint8_t)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server { namespace detail
{
    /// Get the current data for reading
    components::memory_block_data memory_block::get()
    {
        return components::memory_block_data(wrapper_->component_);
    }

    /// Get the current data for reading, use config info for serialization
    components::memory_block_data memory_block::get_config(
        components::memory_block_data const& config)
    {
        return components::memory_block_data(wrapper_->component_, config.data_);
    }

    /// Get the current data for reading
    components::memory_block_data memory_block::checkout() //-V524
    {
        return components::memory_block_data(wrapper_->component_);
    }

    /// Write back data
    void memory_block::checkin(components::memory_block_data const& data)
    {
        // we currently just write back to the memory block
        hpx::actions::manage_object_action_base const& obj =
            wrapper_->component_->get_managing_object();
        obj.assign()(this->get_ptr(), data.get_ptr(), data.get_size());
    }

    naming::gid_type memory_block::clone()
    {
    /// Clone this memory_block
    // FIXME: error code?
        detail::memory_block_header const * rhs = wrapper_->component_.get();
        hpx::actions::manage_object_action_base const& act = this->managing_object_;
        server::memory_block* c = server::memory_block::create(rhs, act);
        naming::gid_type gid = c->get_base_gid();
        if (gid) return gid;

        ::free(c);

        hpx::util::osstream strm;
        strm << "global id " << gid << " is already bound to a different "
                "component instance";
        HPX_THROW_EXCEPTION(hpx::duplicate_component_address,
            "server::detail::create_memory_block",
            hpx::util::osstream_get_string(strm));

        return naming::invalid_gid;
    }
}}}}
