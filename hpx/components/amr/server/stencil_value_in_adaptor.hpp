//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_VALUE_IN_ADAPTOR_OCT_17_2008_0850PM)
#define HPX_COMPONENTS_AMR_STENCIL_VALUE_IN_ADAPTOR_OCT_17_2008_0850PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <hpx/components/amr/server/stencil_value_out_adaptor.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    class stencil_value_in_adaptor 
      : public lcos::eager_future<
            stencil_value_out_adaptor::get_value_action, 
            naming::id_type
        >
    {
    public:
        stencil_value_in_adaptor(applier::applier& appl)
        {}

        // start the asynchronous data acquisition
        void aquire_value(applier::applier& appl) 
        {
            BOOST_ASSERT(gid_);       // must be valid at this point

            this->reset();            // reset the underlying future
            this->apply(appl, gid_);  // asynchronously start the future action 
        }

        // connect this in-port to a data source
        void connect(naming::id_type const& gid)
        {
            gid_ = gid;
        }

    private:
        naming::id_type gid_;
    };

}}}}

#endif

