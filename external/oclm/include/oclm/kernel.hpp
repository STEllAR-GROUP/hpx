
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_KERNEL_HPP
#define OCLM_KERNEL_HPP

#include <string>

#include <oclm/range.hpp>
#include <oclm/function.hpp>

namespace oclm {

    struct kernel
    {
        kernel(program const & p, std::string const & kernel_name)
            : p_(p)
            , kernel_name_(kernel_name)
        {}
    
        template <typename Tag>
        oclm::function operator[](kernel_range<Tag> f)
        {
            kernel_range<tag::collection_> coll;
            coll.set(f);
            return oclm::function(p_, kernel_name_, coll);
        }

        program p_;
        std::string kernel_name_;
    };

}

#endif
