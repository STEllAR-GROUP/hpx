////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_8916D064_084F_48E5_B87A_BDDA4225420A)
#define HPX_8916D064_084F_48E5_B87A_BDDA4225420A

#include <hpx/runtime/agas/traits.hpp>

namespace hpx { namespace agas
{

template <typename Database, typename Key, typename Value>
struct table
{
    typedef typename
        traits::database::map_type<Database, Key, Value>::type
    map_type;

  private:
    map_type table_;

  public:
    table(std::string const& name) : table_()
    { traits::database::connect_table<Database, Key, Value>(table_, name); }
    
    ~table()
    { traits::database::disconnect_table<Database, Key, Value>(table_); }

    // REVIEW: requiring these to return by reference/const reference might be
    // a little too restrictive.
    map_type& get() { return table_; }
    map_type const& get() const { return table_; }
};

}}

#endif // HPX_8916D064_084F_48E5_B87A_BDDA4225420A

