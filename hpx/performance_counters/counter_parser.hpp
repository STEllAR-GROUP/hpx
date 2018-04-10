//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PERFORMANCE_COUNTERS_PARSER_HPP
#define HPX_PERFORMANCE_COUNTERS_PARSER_HPP

#include <hpx/config.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    struct instance_name
    {
        std::string name_;
        std::string index_;
        bool basename_ = false;
    };

    struct instance_elements
    {
        instance_name parent_;
        instance_name child_;
        instance_name subchild_;
    };

    struct path_elements
    {
        std::string object_;
        instance_elements instance_;
        std::string counter_;
        std::string parameters_;
    };

    HPX_API_EXPORT bool parse_counter_name(
        std::string const& name, path_elements& elements);
}}

#endif /*HPX_PERFORMANCE_COUNTERS_PARSER_HPP*/
