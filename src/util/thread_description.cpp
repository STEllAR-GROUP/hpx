//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/util/safe_lexical_cast.hpp>

#include <iostream>
#include <sstream>

namespace hpx { namespace util
{
    std::ostream& operator<<(std::ostream& os, thread_description const& d)
    {
        if (d.kind() == thread_description::data_type_description)
        {
            os << d.get_description();
        }
        else
        {
            HPX_ASSERT(d.kind() == thread_description::data_type_address);
            os << d.get_address();
        }
        return os;
    }

    std::string as_string(thread_description const& desc)
    {
        if (desc.kind() == util::thread_description::data_type_description)
            return desc.get_description();

        std::stringstream strm;
        strm << "address: 0x" << std::hex
                << util::safe_lexical_cast<std::string>(
                    desc.get_address());
        return strm.str();
    }
}}
