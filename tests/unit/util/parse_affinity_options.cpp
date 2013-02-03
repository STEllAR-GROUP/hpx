//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace test
{
    using hpx::threads::detail::spec_type;

    struct data_good
    {
        std::string option_;
        spec_type thread;
        spec_type socket;
        spec_type numanode;
        spec_type core;
        spec_type pu;
    };

    data_good data[] =
    {
        { "thread:1=socket:0",
          spec_type(spec_type::thread, 1, 0),
          spec_type(spec_type::socket, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::unknown, 0, 0)
        },
        { "thread:1=socket:0.numanode:0",
          spec_type(spec_type::thread, 1, 0),
          spec_type(spec_type::socket, 0, 0),
          spec_type(spec_type::numanode, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::unknown, 0, 0)
        },
        { "thread:1=socket:0.numanode:0.core:0",
          spec_type(spec_type::thread, 1, 0),
          spec_type(spec_type::socket, 0, 0),
          spec_type(spec_type::numanode, 0, 0),
          spec_type(spec_type::core, 0, 0),
          spec_type(spec_type::unknown, 0, 0)
        },
        { "thread:1=socket:0.numanode:0.core:0.pu:0",
          spec_type(spec_type::thread, 1, 0),
          spec_type(spec_type::socket, 0, 0),
          spec_type(spec_type::numanode, 0, 0),
          spec_type(spec_type::core, 0, 0),
          spec_type(spec_type::pu, 0, 0)
        },
        { "thread:1=socket:0.numanode:0.pu:0",
          spec_type(spec_type::thread, 1, 0),
          spec_type(spec_type::socket, 0, 0),
          spec_type(spec_type::numanode, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::pu, 0, 0)
        },
        { "thread:1=socket:0.core:0",
          spec_type(spec_type::thread, 1, 0),
          spec_type(spec_type::socket, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::core, 0, 0),
          spec_type(spec_type::unknown, 0, 0)
        },
        { "thread:1=socket:0.core:0.pu:0",
          spec_type(spec_type::thread, 1, 0),
          spec_type(spec_type::socket, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::core, 0, 0),
          spec_type(spec_type::pu, 0, 0)
        },
        { "thread:1=socket:0.pu:0",
          spec_type(spec_type::thread, 1, 0),
          spec_type(spec_type::socket, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::pu, 0, 0)
        },

        { "thread:1=numanode:0",
          spec_type(spec_type::thread, 1, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::numanode, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::unknown, 0, 0)
        },
        { "thread:1=numanode:0.core:0",
          spec_type(spec_type::thread, 1, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::numanode, 0, 0),
          spec_type(spec_type::core, 0, 0),
          spec_type(spec_type::unknown, 0, 0)
        },
        { "thread:1=numanode:0.core:0.pu:0",
          spec_type(spec_type::thread, 1, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::numanode, 0, 0),
          spec_type(spec_type::core, 0, 0),
          spec_type(spec_type::pu, 0, 0)
        },
        { "thread:1=numanode:0.pu:0",
          spec_type(spec_type::thread, 1, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::numanode, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::pu, 0, 0)
        },

        { "thread:1=core:3",
          spec_type(spec_type::thread, 1, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::core, 3, 0),
          spec_type(spec_type::unknown, 0, 0)
        },
        { "thread:1=core:0.pu:2",
          spec_type(spec_type::thread, 1, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::core, 0, 0),
          spec_type(spec_type::pu, 2, 0)
        },

        { "thread:1=pu:2",
          spec_type(spec_type::thread, 1, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::pu, 2, 0)
        },

        { "thread:2=core:all.pu:1",
          spec_type(spec_type::thread, 2, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::core, ~0x0ul, 0),
          spec_type(spec_type::pu, 1, 0)
        },
        { "t:0-3=c:0-3.p:1",
          spec_type(spec_type::thread, 0, 3),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::unknown, 0, 0),
          spec_type(spec_type::core, 0, 3),
          spec_type(spec_type::pu, 1, 0)
        },
        { "" }
    };

    void good()
    {
        for (data_good* t = data; !t->option_.empty(); ++t)
        {
            hpx::threads::detail::mappings_type mappings;
            HPX_TEST(hpx::threads::detail::parse_mappings(t->option_, mappings));
            HPX_TEST(mappings.size() == 1);
            if (mappings.size() == 1) {
                HPX_TEST(t->thread == mappings[0].first);
                HPX_TEST(mappings[0].second.size() == 4);
                if (mappings[0].second.size() == 4) {
                    HPX_TEST(t->socket == mappings[0].second[0]);
                    HPX_TEST(t->numanode == mappings[0].second[1]);
                    HPX_TEST(t->core == mappings[0].second[2]);
                    HPX_TEST(t->pu == mappings[0].second[3]);
                }
            }
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
            HPX_TEST(!hpx::threads::parse_affinity_options(t, affinities));
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        test::good();
        test::bad();
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST(0 == hpx::init(argc, argv));
    return hpx::util::report_errors();
}

