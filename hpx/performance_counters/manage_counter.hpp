////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach and Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_8B1A4443_7D95_4C0D_9970_7CEA4D049608)
#define HPX_8B1A4443_7D95_4C0D_9970_7CEA4D049608

#include <hpx/exception.hpp>
#include <hpx/performance_counters/counters.hpp>

namespace hpx { namespace performance_counters
{
    struct manage_counter
    {
        manage_counter() : counter_(naming::invalid_id) {} 

        ~manage_counter()
        {
            uninstall();
        }

        counter_status install(std::string const& name,
                               boost::function<boost::int64_t()> const& f,
                               error_code& ec = throws)
        {
            BOOST_ASSERT(!counter_);
            info_.fullname_ = name;
            return add_counter(info_, f, counter_, ec);
        }

        void uninstall()
        {
            if (counter_)
            {
                error_code ec;
                remove_counter(info_, counter_, ec);
                counter_ = naming::invalid_id;
            }
        }

      private:
        counter_info info_;
        naming::id_type counter_;
    };
}}

#endif // HPX_8B1A4443_7D95_4C0D_9970_7CEA4D049608

