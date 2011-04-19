////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_EFFD6895_F015_4275_BB17_C3959D5ED258)
#define HPX_EFFD6895_F015_4275_BB17_C3959D5ED258

#include <string>

namespace hpx { namespace agas { namespace traits { namespace database 
{

// MPL metafunction
template <typename Database, typename Key, typename Value,
          typename Enable = void>
struct map_type;

// MPL metafunction
template <typename Database, typename Enable = void>
struct mutex_type;

// Spirit-style CP
template <typename Database, typename Enable = void>
struct name_hook;

template <typename Database>
inline typename name_hook<Database>::result_type name();

// Spirit-style CP
template <typename Database, typename Key, typename Value,
          typename Enable = void>
struct connect_table_hook;

template <typename Database, typename Key, typename Value, typename Map>
inline void connect_table(Map&, std::string const&);

// Spirit-style CP
template <typename Database, typename Key, typename Value,
          typename Enable = void>
struct disconnect_table_hook;

template <typename Database, typename Key, typename Value, typename Map>
inline void disconnect_table(Map&);

}}}}

#endif // HPX_EFFD6895_F015_4275_BB17_C3959D5ED258

