//  Copyright (c) 2005-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if!defined(HPX_INIT_INI_DATA_SEP_26_2008_0344PM)
#define HPX_INIT_INI_DATA_SEP_26_2008_0344PM

#include <string>

#include <hpx/hpx_fwd.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    bool handle_ini_file (section& ini, std::string const& loc);
    bool handle_ini_file_env (section& ini, char const* env_var,
        char const* file_suffix = NULL);

    ///////////////////////////////////////////////////////////////////////////
    // read system and user specified ini files
    //
    // returns true if at least one alternative location has been read
    // successfully
    bool init_ini_data_base(section& ini, std::string const& hpx_ini_file = "");

    ///////////////////////////////////////////////////////////////////////////
    // global function to read component ini information
    void merge_component_inis(section& ini);

    ///////////////////////////////////////////////////////////////////////////
    // iterate over all shared libraries in the given directory and construct
    // default ini settings assuming all of those are components
    void init_ini_data_default(std::string const& libs, section& ini);

}}

#endif

