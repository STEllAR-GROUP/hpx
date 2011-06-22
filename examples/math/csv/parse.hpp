////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_9DA8089B_F928_4260_BF41_20325BE129FA)
#define HPX_9DA8089B_F928_4260_BF41_20325BE129FA

#include <hpx/config.hpp>

#include <string>
#include <vector>

namespace hpx { namespace math { namespace csv
{

enum parse_result
{
    path_does_not_exist,
    path_is_directory,
    parse_succeeded,
    parse_failed
};

typedef std::vector<std::vector<double> > ast;

HPX_EXPORT parse_result parse(std::string const& filename, ast& a);

}}}

#endif // HPX_9DA8089B_F928_4260_BF41_20325BE129FA

