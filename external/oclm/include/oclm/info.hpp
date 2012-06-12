
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_INFO_HPP
#define OCLM_INFO_HPP

#include <oclm/config.hpp>

namespace oclm
{
    template <
        typename T
      , typename Result
      , cl_int (*F)(T, unsigned, ::size_t, void *, ::size_t *)
      , int Name
    >
    struct info
    {
        typedef Result result_type;

        static cl_int get(T t, ::size_t size, void * value, ::size_t * size_ret)
        {
            return F(t, Name, size, value, size_ret);
        }
    };
}

#endif
