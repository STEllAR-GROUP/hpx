// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace components
{
    template <typename T>
    inline HPX_STD_TUPLE<access_memory_block<T> , access_memory_block<T> >
    get_memory_block_async(naming::id_type const& g0 , naming::id_type const& g1)
    {
        return lcos::wait(stubs::memory_block::get_async(g0) , stubs::memory_block::get_async(g1));
    }
}}
namespace hpx { namespace components
{
    template <typename T>
    inline HPX_STD_TUPLE<access_memory_block<T> , access_memory_block<T> , access_memory_block<T> >
    get_memory_block_async(naming::id_type const& g0 , naming::id_type const& g1 , naming::id_type const& g2)
    {
        return lcos::wait(stubs::memory_block::get_async(g0) , stubs::memory_block::get_async(g1) , stubs::memory_block::get_async(g2));
    }
}}
namespace hpx { namespace components
{
    template <typename T>
    inline HPX_STD_TUPLE<access_memory_block<T> , access_memory_block<T> , access_memory_block<T> , access_memory_block<T> >
    get_memory_block_async(naming::id_type const& g0 , naming::id_type const& g1 , naming::id_type const& g2 , naming::id_type const& g3)
    {
        return lcos::wait(stubs::memory_block::get_async(g0) , stubs::memory_block::get_async(g1) , stubs::memory_block::get_async(g2) , stubs::memory_block::get_async(g3));
    }
}}
namespace hpx { namespace components
{
    template <typename T>
    inline HPX_STD_TUPLE<access_memory_block<T> , access_memory_block<T> , access_memory_block<T> , access_memory_block<T> , access_memory_block<T> >
    get_memory_block_async(naming::id_type const& g0 , naming::id_type const& g1 , naming::id_type const& g2 , naming::id_type const& g3 , naming::id_type const& g4)
    {
        return lcos::wait(stubs::memory_block::get_async(g0) , stubs::memory_block::get_async(g1) , stubs::memory_block::get_async(g2) , stubs::memory_block::get_async(g3) , stubs::memory_block::get_async(g4));
    }
}}
