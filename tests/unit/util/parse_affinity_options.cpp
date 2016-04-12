//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>

#include <hpx/include/threads.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/assign/std/vector.hpp>

#include <algorithm>
#include <iostream>
#include <string>

// The affinity masks this test is verifying the results against are specific
// to a particular machine. If you enable this option you might see a lot of
// test failures, which is expected.
// The bit masks in the tests below are assuming a 12 core system (with
// hyper threading), with 2 NUMA nodes (2 sockets), 6 cores each.
//#define VERIFY_AFFINITY_MASKS

///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_HWLOC)
namespace hpx { namespace threads { namespace detail
{
    std::ostream& operator<<(std::ostream& os, spec_type const& data)
    {
        os << spec_type::type_name(data.type_);
        for (std::size_t i : data.index_bounds_)
        {
            os  << "," << i;
        }
        return os;
    }
}}}

namespace test
{
    using hpx::threads::detail::spec_type;

    struct data_good_thread
    {
        spec_type thread;
        spec_type socket;
        spec_type core;
        spec_type pu;
    };

    struct data_good
    {
        std::string option_;
        data_good_thread t[2];
        boost::uint64_t masks[2];
    };

//  Test cases implemented below:
//
//   thread:0-1=socket:0
//   thread:0-1=socket:0-1
//   thread:0-1=numanode:0
//   thread:0-1=numanode:0-1
//   thread:0-1=core:0
//   thread:0-1=core:0-1
//   thread:0-1=core:0.pu:0
//   thread:0-1=core:0.pu:0-1
//   thread:0-1=pu:0
//   thread:0-1=pu:0-1
//   thread:0-1=socket:0.core:0
//   thread:0-1=socket:1.core:0-1
//   thread:0-1=numanode:0.core:0
//   thread:0-1=numanode:1.core:0-1
//   thread:0-1=socket:1.core:1.pu:0
//   thread:0-1=socket:1.core:1.pu:0-1
//   thread:0-1=numanode:1.core:1.pu:0
//   thread:0-1=numanode:1.core:1.pu:0-1
//   thread:0-1=socket:1.core:0-1.pu:1
//   thread:0-1=numanode:1.core:0-1.pu:1
//   thread:0-1=socket:0-1.core:1.pu:1
//   thread:0-1=numanode:0-1.core:1.pu:1
//   thread:0-1=socket:0-1.pu:0
//   thread:0-1=numanode:0-1.pu:0
//   thread:0-1=socket:0.pu:0
//   thread:0-1=socket:0.pu:0-1
//   thread:0-1=numanode:0.pu:0
//   thread:0-1=numanode:0.pu:0-1
//   thread:0-1=socket:0.core:all.pu:0

    data_good data[] =
    {
        {   "thread:0=socket:0;thread:1=socket:0",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }
            },
            { 0x000fff, 0x000fff }
        },
        {   "thread:0-1=socket:0",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x000fff, 0x000fff }
        },
        {   "thread:0,1=socket:0",
            {
                {
                    spec_type(spec_type::thread, 0, 1),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x000fff, 0x000fff }
        },

        {   "thread:0=socket:0;thread:1=socket:1",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::socket, 1, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }
            },
            { 0x000fff, 0xfff000 }
        },
        {   "thread:0-1=socket:0-1",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::socket, 0, -1),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x000fff, 0xfff000 }
        },
        {   "thread:0,1=socket:0-1",
            {
                {
                    spec_type(spec_type::thread, 0, 1),
                    spec_type(spec_type::socket, 0, -1),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x000fff, 0xfff000 }
        },

        {   "thread:0=numanode:0;thread:1=numanode:0",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }
            },
            { 0x000fff, 0x000fff }
        },
        {   "thread:0-1=numanode:0",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x000fff, 0x000fff }
        },
        {   "thread:0,1=numanode:0",
            {
                {
                    spec_type(spec_type::thread, 0, 1),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x000fff, 0x000fff }
        },

        {   "thread:0=numanode:0;thread:1=numanode:1",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::numanode, 1, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }
            },
            { 0x000fff, 0xfff000 }
        },
        {   "thread:0-1=numanode:0-1",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::numanode, 0, -1),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x000fff, 0xfff000 }
        },
        {   "thread:0,1=numanode:0,1",
            {
                {
                    spec_type(spec_type::thread, 0, 1),
                    spec_type(spec_type::numanode, 0, 1),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x000fff, 0xfff000 }
        },

        {   "thread:0=core:0;thread:1=core:0",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::unknown)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::unknown)
                }
            },
            { 0x000003, 0x000003 }
        },
        {   "thread:0-1=core:0",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x000003, 0x000003 }
        },
        {   "thread:0,1=core:0",
            {
                {
                    spec_type(spec_type::thread, 0, 1),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x000003, 0x000003 }
        },

        {   "thread:0=core:0;thread:1=core:1",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::unknown)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::unknown)
                }
            },
            { 0x000003, 0x00000c }
        },
        {   "thread:0-1=core:0-1",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::core, 0, -1),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x000003, 0x00000c }
        },
        {   "thread:0,1=core:0,1",
            {
                {
                    spec_type(spec_type::thread, 0, 1),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::core, 0, 1),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x000003, 0x00000c }
        },

        {   "thread:0=core:1.pu:0;thread:1=core:1.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 0, 0)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 0, 0)
                }
            },
            { 0x000004, 0x000004 }
        },
        {   "thread:0-1=core:1.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 0, 0)
                }, data_good_thread()
            },
            { 0x000004, 0x000004 }
        },

        {   "thread:0=core:1.pu:0;thread:1=core:1.pu:1",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 0, 0)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 1, 0)
                }
            },
            { 0x000004, 0x000008 }
        },
        {   "thread:0-1=core:1.pu:0-1",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 0, -1)
                }, data_good_thread()
            },
            { 0x000004, 0x000008 }
        },

        {   "thread:0=pu:0;thread:1=pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                }
            },
            { 0x000001, 0x000001 }
        },
        {   "thread:0-1=pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                }, data_good_thread()
            },
            { 0x000001, 0x000001 }
        },

        {   "thread:0=pu:0;thread:1=pu:1",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 1, 0)
                }
            },
            { 0x000001, 0x000002 }
        },
        {   "thread:0-1=pu:0-1",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, -1)
                }, data_good_thread()
            },
            { 0x000001, 0x000002 }
        },

        {   "thread:0=socket:0.core:0;thread:1=socket:0.core:0",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::unknown)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::unknown)
                }
            },
            { 0x000003, 0x000003 }
        },
        {   "thread:0-1=socket:0.core:0",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x000003, 0x000003 }
        },

        {   "thread:0=socket:1.core:0;thread:1=socket:1.core:1",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::socket, 1, 0),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::unknown)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::socket, 1, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::unknown)
                }
            },
            { 0x003000, 0x00c000 }
        },
        {   "thread:0-1=socket:1.core:0-1",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::socket, 1, 0),
                    spec_type(spec_type::core, 0, -1),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x003000, 0x00c000 }
        },

        {   "thread:0=numanode:0.core:0;thread:1=numanode:0.core:0",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::unknown)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::unknown)
                }
            },
            { 0x000003, 0x000003 }
        },
        {   "thread:0-1=numanode:0.core:0",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x000003, 0x000003 }
        },

        {   "thread:0=numanode:1.core:0;thread:1=numanode:1.core:1",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::numanode, 1, 0),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::unknown)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::numanode, 1, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::unknown)
                }
            },
            { 0x003000, 0x00c000 }
        },
        {   "thread:0-1=numanode:1.core:0-1",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::numanode, 1, 0),
                    spec_type(spec_type::core, 0, -1),
                    spec_type(spec_type::unknown)
                }, data_good_thread()
            },
            { 0x003000, 0x00c000 }
        },

        {   "thread:0=socket:1.core:0.pu:1;thread:1=socket:1.core:1.pu:1",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::socket, 1, 0),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::pu, 1, 0)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::socket, 1, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 1, 0)
                }
            },
            { 0x002000, 0x008000 }
        },
        {   "thread:0-1=socket:1.core:0-1.pu:1",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::socket, 1, 0),
                    spec_type(spec_type::core, 0, -1),
                    spec_type(spec_type::pu, 1, 0)
                }, data_good_thread()
            },
            { 0x002000, 0x008000 }
        },

        {   "thread:0=socket:1.core:1.pu:0;thread:1=socket:1.core:1.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::socket, 1, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 0, 0)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::socket, 1, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 0, 0)
                }
            },
            { 0x004000, 0x004000 }
        },
        {   "thread:0-1=socket:1.core:1.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::socket, 1, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 0, 0)
                }, data_good_thread()
            },
            { 0x004000, 0x004000 }
        },

        {   "thread:0=socket:1.core:1.pu:0;thread:1=socket:1.core:1.pu:1",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::socket, 1, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 0, 0)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::socket, 1, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 1, 0)
                }
            },
            { 0x004000, 0x008000 }
        },
        {   "thread:0-1=socket:1.core:1.pu:0-1",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::socket, 1, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 0, -1)
                }, data_good_thread()
            },
            { 0x004000, 0x008000 }
        },

        {   "thread:0=numanode:1.core:1.pu:0;thread:1=numanode:1.core:1.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::numanode, 1, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 0, 0)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::numanode, 1, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 0, 0)
                }
            },
            { 0x004000, 0x004000 }
        },
        {   "thread:0-1=numanode:1.core:1.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::numanode, 1, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 0, 0)
                }, data_good_thread()
            },
            { 0x004000, 0x004000 }
        },

        {   "thread:0=numanode:1.core:0.pu:1;thread:1=numanode:1.core:1.pu:1",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::numanode, 1, 0),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::pu, 1, 0)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::numanode, 1, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 1, 0)
                }
            },
            { 0x002000, 0x008000 }
        },
        {   "thread:0-1=numanode:1.core:0-1.pu:1",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::numanode, 1, 0),
                    spec_type(spec_type::core, 0, -1),
                    spec_type(spec_type::pu, 1, 0)
                }, data_good_thread()
            },
            { 0x002000, 0x008000 }
        },

        {   "thread:0-1=socket:0.core:all.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::core, spec_type::all_entities(), 0),
                    spec_type(spec_type::pu, 0, 0)
                }, data_good_thread()
            },
            { 0x000001, 0x000004 }
        },

        {   "thread:0=socket:0.core:1.pu:1;thread:1=socket:1.core:1.pu:1",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 1, 0)
                },
                {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::socket, 1, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 1, 0)
                }
            },
            { 0x000008, 0x008000 }
        },
        {   "thread:0-1=socket:0-1.core:1.pu:1",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::socket, 0, -1),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 1, 0)
                }, data_good_thread()
            },
            { 0x000008, 0x008000 }
        },

        {   "thread:0=numanode:0.core:1.pu:1;thread:1=numanode:1.core:1.pu:1",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 1, 0)
                },
                {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::numanode, 1, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 1, 0)
                }
            },
            { 0x000008, 0x008000 }
        },
        {   "thread:0-1=numanode:0-1.core:1.pu:1",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::numanode, 0, -1),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::pu, 1, 0)
                }, data_good_thread()
            },
            { 0x000008, 0x008000 }
        },

        {   "thread:0=socket:0.pu:0;thread:1=socket:1.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                },
                {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::socket, 1, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                }
            },
            { 0x000001, 0x001000 }
        },
        {   "thread:0-1=socket:0-1.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::socket, 0, -1),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                }, data_good_thread()
            },
            { 0x000001, 0x001000 }
        },

        {   "thread:0=numanode:0.pu:0;thread:1=numanode:1.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                },
                {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::numanode, 1, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                }
            },
            { 0x000001, 0x001000 }
        },
        {   "thread:0-1=numanode:0-1.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::numanode, 0, -1),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                }, data_good_thread()
            },
            { 0x000001, 0x001000 }
        },

        {   "thread:0=socket:0.pu:0;thread:1=socket:0.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                },
                {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                }
            },
            { 0x000001, 0x000001 }
        },
        {   "thread:0-1=socket:0.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                }, data_good_thread()
            },
            { 0x000001, 0x000001 }
        },

        {   "thread:0=socket:0.pu:0;thread:1=socket:0.pu:1",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                },
                {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 1, 0)
                }
            },
            { 0x000001, 0x000002 }
        },
        {   "thread:0-1=socket:0.pu:0-1",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, -1)
                }, data_good_thread()
            },
            { 0x000001, 0x000002 }
        },

        {   "thread:0=numanode:0.pu:0;thread:1=numanode:0.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                },
                {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                }
            },
            { 0x000001, 0x000001 }
        },
        {   "thread:0-1=numanode:0.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                }, data_good_thread()
            },
            { 0x000001, 0x000001 }
        },

        {   "thread:0=numanode:0.pu:0;thread:1=numanode:0.pu:1",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, 0)
                },
                {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 1, 0)
                }
            },
            { 0x000001, 0x000002 }
        },
        {   "thread:0-1=numanode:0.pu:0-1",
            {
                {
                    spec_type(spec_type::thread, 0, -1),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::unknown),
                    spec_type(spec_type::pu, 0, -1)
                }, data_good_thread()
            },
            { 0x000001, 0x000002 }
        },

        { "", {data_good_thread(), data_good_thread()}, {0,0} }
    };

    void good_testing(data_good const* t, char const* const options)
    {
        hpx::threads::detail::mappings_type mappings;
        hpx::error_code ec;
        hpx::threads::detail::parse_mappings(options, mappings, ec);
        HPX_TEST(!ec);

        int i = 0;

        HPX_TEST_EQ(mappings.which(), 1);
        if (mappings.which() == 1)
        {
            hpx::threads::detail::mappings_spec_type mappings_specs(
                boost::get<hpx::threads::detail::mappings_spec_type>(mappings));

            for (hpx::threads::detail::full_mapping_type const& m : mappings_specs)
            {
                HPX_TEST_EQ(t->t[i].thread, m.first);
                HPX_TEST_EQ(m.second.size(), 3u);
                if (m.second.size() == 3u) {
                    HPX_TEST_EQ(t->t[i].socket, m.second[0]);
                    HPX_TEST_EQ(t->t[i].core, m.second[1]);
                    HPX_TEST_EQ(t->t[i].pu, m.second[2]);
                }
                ++i;
            }
        }

#if defined(VERIFY_AFFINITY_MASKS)
        std::vector<hpx::threads::mask_type> affinities;
        affinities.resize(hpx::get_os_thread_count(), 0);
        hpx::threads::parse_affinity_options(t->option_, affinities, ec);
        HPX_TEST(!ec);
        HPX_TEST_EQ(affinities.size(), 2u);
        HPX_TEST_EQ(std::count(affinities.begin(), affinities.end(), 0), 0);

        i = 0;
        for (hpx::threads::mask_type m : affinities)
        {
            HPX_TEST_EQ(t->masks[i], m);
            ++i;
        }
#endif
    }

    std::string replace_all(std::string str, char const* const what,
        char const* const with)
    {
        std::string::size_type p = str.find(what);
        if (p != std::string::npos) {
            std::size_t len = std::strlen(what);
            do {
                str.replace(p, len, with);
                p = str.find(what, p+len);
            } while (p != std::string::npos);
        }
        return str;
    }

    std::string shorten_options(std::string str)
    {
        str = replace_all(str, "thread", "t");
        str = replace_all(str, "socket", "s");
        str = replace_all(str, "numanode", "n");
        str = replace_all(str, "core", "c");
        return replace_all(str, "pu", "p");
    }

    void good()
    {
        for (data_good* t = data; !t->option_.empty(); ++t)
        {
            // test full length options
            good_testing(t, t->option_.c_str());

            // test shortened options
            std::string shortened_options(shorten_options(t->option_));
            good_testing(t, shortened_options.c_str());
        }
    }

    char const* const data_bad[] =
    {
        // wrong sequence
        "thread:0=pu:0.socket:0",
        "thread:0=pu:0.numanode:0",
        "thread:0=pu:0.core:0",
        "thread:0=core:0.socket:0",
        "thread:0=core:0.numanode:0",
        "thread:0=numanode:0.socket:0",
        // duplicates
        "thread:0=socket:0.socket:0",
        "thread:0=numanode:0.numanode:0",
        "thread:0=core:0.core:0",
        "thread:0=pu:0.pu:0",
        "thread:1=socket:0.numanode:0",
        // empty
        "thread:0=socket",
        "thread:0=numanode",
        "thread:0=core",
        "thread:0=pu",
        "thread=",
        "socket:0",
        "numanode:0",
        "core:0",
        "pu:0",
        NULL
    };

    void bad()
    {
        int i = 0;
        hpx::error_code ec;
        for (char const* t = data_bad[0]; NULL != t; t = data_bad[++i])
        {
            std::vector<hpx::threads::mask_type> affinities;
            hpx::threads::parse_affinity_options(t, affinities, ec);
            HPX_TEST(ec);
        }
    }
}
#endif

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
#if defined(HPX_HAVE_HWLOC)
    {
        test::good();
        test::bad();
    }
#endif

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // We force this test to use 2 threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=2";

    // Initialize and run HPX
    HPX_TEST(0 == hpx::init(argc, argv, cfg));
    return hpx::util::report_errors();
}

