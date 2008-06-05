//  Copyright (c) 2007-2008 Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLIER_JUN_03_2008_0438PM)
#define HPX_APPLIER_APPLIER_JUN_03_2008_0438PM

#include <hpx/naming.hpp>
#include <hpx/parcelset.hpp>
#include <hpx/threadmanager.hpp>

namespace hpx { namespace applier
{
    class applier
    {
    public:
        // constructor
        applier(naming::resolver_client& dgas_c, threadmanager::threadmanager& tm, 
            parcelset::parcelhandler &ph)
            : dgas_client_(dgas_c), thread_manager_(tm), parcel_handler_(ph)
        {
        }

        // destructor
        ~applier()
        {
        }

        // Invoked by a running PX-thread to apply an action to any resource
        template <typename Action>
        void apply (naming::id_type gid);

        template <typename Action, typename Arg0>
        void apply (naming::id_type gid, Arg0 const& arg0);

        template <typename Action, typename Arg0, typename Arg1>
        void apply (naming::id_type gid, Arg0 const& arg0, Arg1 const& arg1);

    private:
        naming::resolver_client& dgas_client_;
        threadmanager::threadmanager& thread_manager_;
        parcelset::parcelhandler& parcel_handler_;
    };
}}

#endif
