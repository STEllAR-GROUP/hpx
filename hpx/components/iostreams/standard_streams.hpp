////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_8F5A7F0B_E4CE_422C_B58A_2AEC43AD2057)
#define HPX_8F5A7F0B_E4CE_422C_B58A_2AEC43AD2057

#include <hpx/config.hpp>
#include <hpx/components/iostreams/export_definitions.hpp>
#include <hpx/components/iostreams/lazy_ostream.hpp>

namespace hpx
{
    namespace iostreams
    {
        HPX_IOSTREAMS_EXPORT void create_cout();
        HPX_IOSTREAMS_EXPORT void create_cerr();

        HPX_IOSTREAMS_EXPORT lazy_ostream& cout();
        HPX_IOSTREAMS_EXPORT lazy_ostream& cerr();
    }

    ///////////////////////////////////////////////////////////////////////////
    struct cout_wrapper {};
    struct cerr_wrapper {};

    HPX_IOSTREAMS_EXPORT extern cout_wrapper cout;
    HPX_IOSTREAMS_EXPORT extern cerr_wrapper cerr;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline cout_wrapper& operator<< (cout_wrapper& s, T const& t)
    {
        iostreams::cout() << t;
        return s;
    }

    template <typename T>
    inline cerr_wrapper& operator<< (cerr_wrapper& s, T const& t)
    {
        iostreams::cerr() << t;
        return s;
    }
}

#endif // HPX_8F5A7F0B_E4CE_422C_B58A_2AEC43AD2057

