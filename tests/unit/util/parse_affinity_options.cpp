//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>

#if defined(HPX_HAVE_HWLOC)
#include <hpx/include/threads.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/assign/std/vector.hpp>

#include <iostream>
#include <algorithm>

// The affinity masks this test is verifying the results against are specific 
// to a particular machine. If you enable this option you might see a lot of 
// test failures, which is expected.
// #define VERIFY_AFFINITY_MASKS

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace detail
{
    std::ostream& operator<<(std::ostream& os, spec_type const& data)
    {
        os << spec_type::type_name(data.type_)
           << "," << data.index_min_ << "," << data.index_max_;
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
        hpx::threads::mask_type masks[2];
    };

    data_good data[] =
    {
        {   "thread:0=socket:0;thread:1=socket:0",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::unknown, 0, 0),
                    spec_type(spec_type::unknown, 0, 0)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::unknown, 0, 0),
                    spec_type(spec_type::unknown, 0, 0)
                }
            },
            { 0xfff, 0xfff }
        },
        {   "thread:0-1=socket:0",
            {
                {
                    spec_type(spec_type::thread, 0, 1),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::unknown, 0, 0),
                    spec_type(spec_type::unknown, 0, 0)
                }
            },
            { 0xfff, 0xfff }
        },
        {   "thread:0=numanode:0;thread:1=numanode:0",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::unknown, 0, 0),
                    spec_type(spec_type::unknown, 0, 0)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::unknown, 0, 0),
                    spec_type(spec_type::unknown, 0, 0)
                }
            },
            { 0xfff, 0xfff }
        },
        {   "thread:0-1=numanode:0",
            {
                {
                    spec_type(spec_type::thread, 0, 1),
                    spec_type(spec_type::numanode, 0, 0),
                    spec_type(spec_type::unknown, 0, 0),
                    spec_type(spec_type::unknown, 0, 0)
                }
            },
            { 0xfff, 0xfff }
        },
        {   "thread:0=socket:0.core:0;thread:1=socket:0.core:1",
            {
                {
                    spec_type(spec_type::thread, 0, 0),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::core, 0, 0),
                    spec_type(spec_type::unknown, 0, 0)
                }, {
                    spec_type(spec_type::thread, 1, 0),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::core, 1, 0),
                    spec_type(spec_type::unknown, 0, 0)
                }
            },
            { 0x003, 0x00c }
        },
        {   "thread:0-1=socket:0.core:0-1",
            {
                {
                    spec_type(spec_type::thread, 0, 1),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::core, 0, 1),
                    spec_type(spec_type::unknown, 0, 0)
                }
            },
            { 0x003, 0x00c }
        },
        {   "thread:0-1=socket:0.core:all.pu:0",
            {
                {
                    spec_type(spec_type::thread, 0, 1),
                    spec_type(spec_type::socket, 0, 0),
                    spec_type(spec_type::core, ~0x0ul, 0),
                    spec_type(spec_type::pu, 0, 0)
                }
            },
            { 0x001, 0x008 }
        },

//         { "thread:0-1=socket:0.core:0-1",
//           spec_type(spec_type::thread, 0, 1),
//           spec_type(spec_type::socket, 0, 0),
//           spec_type(spec_type::core, 0, 1),
//           spec_type(spec_type::unknown, 0, 0)
//         },
//         { "thread:1=numanode:0.core:0",
//           spec_type(spec_type::thread, 1, 0),
//           spec_type(spec_type::numanode, 0, 0),
//           spec_type(spec_type::core, 0, 0),
//           spec_type(spec_type::unknown, 0, 0)
//         },
//         { "thread:1=socket:0.core:0.pu:0",
//           spec_type(spec_type::thread, 1, 0),
//           spec_type(spec_type::socket, 0, 0),
//           spec_type(spec_type::core, 0, 0),
//           spec_type(spec_type::pu, 0, 0)
//         },
//         { "thread:1=numanode:0.core:0.pu:0",
//           spec_type(spec_type::thread, 1, 0),
//           spec_type(spec_type::numanode, 0, 0),
//           spec_type(spec_type::core, 0, 0),
//           spec_type(spec_type::pu, 0, 0)
//         },
//         { "thread:1=socket:0.pu:0",
//           spec_type(spec_type::thread, 1, 0),
//           spec_type(spec_type::socket, 0, 0),
//           spec_type(spec_type::unknown, 0, 0),
//           spec_type(spec_type::pu, 0, 0)
//         },
//         { "thread:1=numanode:0.pu:0",
//           spec_type(spec_type::thread, 1, 0),
//           spec_type(spec_type::numanode, 0, 0),
//           spec_type(spec_type::unknown, 0, 0),
//           spec_type(spec_type::pu, 0, 0)
//         },
//
//         { "thread:1=core:3",
//           spec_type(spec_type::thread, 1, 0),
//           spec_type(spec_type::unknown, 0, 0),
//           spec_type(spec_type::core, 3, 0),
//           spec_type(spec_type::unknown, 0, 0)
//         },
//         { "thread:1=core:0.pu:2",
//           spec_type(spec_type::thread, 1, 0),
//           spec_type(spec_type::unknown, 0, 0),
//           spec_type(spec_type::core, 0, 0),
//           spec_type(spec_type::pu, 2, 0)
//         },
//
//         { "thread:1=pu:2",
//           spec_type(spec_type::thread, 1, 0),
//           spec_type(spec_type::unknown, 0, 0),
//           spec_type(spec_type::unknown, 0, 0),
//           spec_type(spec_type::pu, 2, 0)
//         },
//
//         { "thread:2=core:all.pu:1",
//           spec_type(spec_type::thread, 2, 0),
//           spec_type(spec_type::unknown, 0, 0),
//           spec_type(spec_type::core, ~0x0ul, 0),
//           spec_type(spec_type::pu, 1, 0)
//         },
//         { "t:0-3=c:0-3.p:1",
//           spec_type(spec_type::thread, 0, 3),
//           spec_type(spec_type::unknown, 0, 0),
//           spec_type(spec_type::core, 0, 3),
//           spec_type(spec_type::pu, 1, 0)
//         },
        { "" }
    };

    void good()
    {
        for (data_good* t = data; !t->option_.empty(); ++t)
        {
            hpx::threads::detail::mappings_type mappings;
            hpx::error_code ec;
            hpx::threads::detail::parse_mappings(t->option_, mappings, ec);
            HPX_TEST(!ec);

            int i = 0;
            BOOST_FOREACH(hpx::threads::detail::full_mapping_type const& m, mappings)
            {
                HPX_TEST_EQ(t->t[i].thread, m.first);
                HPX_TEST_EQ(m.second.size(), 3);
                if (m.second.size() == 3) {
                    HPX_TEST_EQ(t->t[i].socket, m.second[0]);
                    HPX_TEST_EQ(t->t[i].core, m.second[1]);
                    HPX_TEST_EQ(t->t[i].pu, m.second[2]);
                }
                ++i;
            }

#if defined(VERIFY_AFFINITY_MASKS)
            std::vector<hpx::threads::mask_type> affinities;
            affinities.resize(hpx::get_os_thread_count(), 0);
            hpx::threads::parse_affinity_options(t->option_, affinities, ec);
            HPX_TEST(!ec);
            HPX_TEST_EQ(affinities.size(), 2);
            HPX_TEST_EQ(std::count(affinities.begin(), affinities.end(), 0), 0);

            i = 0;
            BOOST_FOREACH(hpx::threads::mask_type m, affinities)
            {
                HPX_TEST_EQ(t->masks[i], m);
                ++i;
            }
#endif
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
        for (char const* t = data_bad[0]; NULL != t; t = data_bad[++i])
        {
            std::vector<hpx::threads::mask_type> affinities;
            hpx::error_code ec;
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

