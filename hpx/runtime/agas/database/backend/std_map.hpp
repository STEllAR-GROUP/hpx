////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_314EEA8B_F6A1_4BF3_8BCD_51D3F8B317A5)
#define HPX_314EEA8B_F6A1_4BF3_8BCD_51D3F8B317A5

#include <map>

#include <hpx/runtime/agas/traits.hpp>

namespace hpx { namespace agas 
{

namespace tag { namespace database { struct std_map; }} 

namespace traits { namespace database
{

// REVIEW: Consider using HPX allocator stuff (possibly moot because of malloc
// overloading that we do).
template <typename Key, typename Value>
struct map_type<tag::database::std_map, Key, Value>
{ typedef std::map<Key, Value> type; };

template <>
struct name_hook<tag::database::std_map>
{
    typedef char const* result_type;

    static result_type call()
    { return "std_map"; }
};

}}}}

#endif // HPX_314EEA8B_F6A1_4BF3_8BCD_51D3F8B317A5

