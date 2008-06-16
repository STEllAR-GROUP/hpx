//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_FULLEMPTYSTORE_JUN_16_2008_0128APM)
#define HPX_LCOS_FULLEMPTYSTORE_JUN_16_2008_0128APM

#include <boost/thread.hpp>

namespace hpx { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    class full_empty_entry
    {
    public:
        full_empty_entry()
          : full(true)
        {}
        
    private:
        boost::mutex mtx_;
//         std::queue<> 
        bool full;
    };
    
}}}

#endif

