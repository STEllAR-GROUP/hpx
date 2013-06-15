//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_VALUE_IN_ADAPTOR_OCT_17_2008_0850PM)
#define HPX_COMPONENTS_AMR_STENCIL_VALUE_IN_ADAPTOR_OCT_17_2008_0850PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/lcos/packaged_action.hpp>

#include "stencil_value_out_adaptor.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server
{
    class stencil_value_in_adaptor
      : public lcos::packaged_action<
            stencil_value_out_adaptor::get_value_action, 
            naming::id_type
        >
    {
    public:
        stencil_value_in_adaptor()
          : gid_(naming::invalid_id)
        {}

        // start the asynchronous data acquisition
        void aquire_value()
        {
            BOOST_ASSERT(gid_);       // must be valid at this point

            this->reset();            // reset the underlying future
            this->apply(launch::all, gid_);        // asynchronously start the future action
        }

        // connect this in-port to a data source
        void connect(naming::id_type const& gid)
        {
            gid_ = gid;
        }

        // return whether this input port has been bound to an output port
        bool is_bound() const
        {
            return gid_ != naming::invalid_id;
        }

    private:
        naming::id_type gid_;
    };

}}}}

#endif

