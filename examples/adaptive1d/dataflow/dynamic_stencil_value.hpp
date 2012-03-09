//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_DATAFLOW_STENCIL_VALUE_NOV_02_2011_0506PM)
#define HPX_COMPONENTS_DATAFLOW_STENCIL_VALUE_NOV_02_2011_0506PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/lcos/async.hpp>

#include "stubs/dynamic_stencil_value.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace adaptive1d
{
    /// \class dynamic_stencil_value dynamic_stencil_value.hpp hpx/components/adaptive1d/dynamic_stencil_value.hpp
    class dynamic_stencil_value
      : public client_base<dynamic_stencil_value, adaptive1d::stubs::dynamic_stencil_value >
    {
    private:
        typedef
            client_base<dynamic_stencil_value, adaptive1d::stubs::dynamic_stencil_value >
        base_type;

    public:
        dynamic_stencil_value() {}

        /// Construct a new dynamic_stencil_value instance
        dynamic_stencil_value(naming::id_type const& gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        ///////////////////////////////////////////////////////////////////////
        /// Invokes the time series evolution for this data point using the
        /// data referred to by the parameter \a initial. After finishing
        /// execution it returns a reference to the result as its return value
        /// (parameter \a result)
        lcos::future<naming::id_type> call_async(
            naming::id_type const& initial)
        {
            return this->base_type::call_async(this->gid_, initial);
        }

        naming::id_type call(naming::id_type const& initial)
        {
            return this->base_type::call(this->gid_, initial);
        }

        ///////////////////////////////////////////////////////////////////////
        /// Return the gid's of the output ports associated with this
        /// \a dynamic_stencil_value instance.
        lcos::future<std::vector<naming::id_type> >
        get_output_ports_async()
        {
            return this->base_type::get_output_ports_async(this->gid_);
        }

        std::vector<naming::id_type>
        get_output_ports()
        {
            return this->base_type::get_output_ports(this->gid_);
        }

        ///////////////////////////////////////////////////////////////////////
        /// Connect the destinations given by the provided gid's with the
        /// corresponding input ports associated with this \a dynamic_stencil_value
        /// instance.
        void connect_input_ports(std::vector<naming::id_type> const& gids)
        {
            this->base_type::connect_input_ports(this->gid_, gids);
        }

        ///////////////////////////////////////////////////////////////////////
        /// Set the gid of the component implementing the actual time evolution
        /// functionality
        void set_functional_component(naming::id_type const& functiongid,
            int row, int column, int instencilsize, int outstencilsize,
            double cycle_time,parameter const& par)
        {
            this->base_type::set_functional_component(this->gid_, functiongid,
                row, column, instencilsize, outstencilsize, cycle_time,par);
        }

        ///////////////////////////////////////////////////////////////////////
        /// Subset of set_functional_component functionality
        void start()
        {
            this->base_type::start(this->gid_);
        }
    };

}}}

#endif

