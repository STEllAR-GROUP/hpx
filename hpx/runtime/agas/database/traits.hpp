////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_8BB20BFD_6A13_4434_8E9E_F2312A01A7C3)
#define HPX_8BB20BFD_6A13_4434_8E9E_F2312A01A7C3

#include <hpx/runtime/agas/database/traits_fwd.hpp>

#include <hpx/lcos/mutex.hpp>

namespace hpx { namespace agas { namespace traits { namespace database 
{

template <typename Database, typename Enable>
struct mutex_type 
{ typedef hpx::lcos::mutex type; };

template <typename Database>
inline typename name_hook<Database>::result_type name()
{ return name_hook<Database>::call(); }

template <typename Database, typename Key, typename Value, typename Enable>
struct connect_table_hook
{
    typedef typename map_type<Database, Key, Value>::type map_type;

    static void call(map_type&, std::string const&)
    { /* default is a no-op */ }
};

template <typename Database, typename Key, typename Value, typename Map>
inline void connect_table(Map& table, std::string const& name)
{ connect_table_hook<Database, Key, Value>::call(table, name); } 

template <typename Database, typename Key, typename Value, typename Enable>
struct disconnect_table_hook
{
    typedef typename map_type<Database, Key, Value>::type map_type;

    static void call(map_type&)
    { /* default is a no-op */ }
};

template <typename Database, typename Key, typename Value, typename Map>
inline void disconnect_table(Map& table)
{ disconnect_table_hook<Database, Key, Value>::call(table); } 

}}}}

#endif // HPX_8BB20BFD_6A13_4434_8E9E_F2312A01A7C3

