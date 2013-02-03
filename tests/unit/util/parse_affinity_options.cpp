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
    struct data_good
    {
        std::string option_;
    };

    data_good data[] =
    {
        { "thread:1=socket:0"
        },
        { "thread:1=socket:0.numanode:0"
        },
        { "thread:1=socket:0.numanode:0.core:0"
        },
        { "thread:1=socket:0.numanode:0.core:0.pu:0"
        },
        { "thread:1=socket:0.numanode:0.pu:0"
        },
        { "thread:1=socket:0.core:0"
        },
        { "thread:1=socket:0.core:0.pu:0"
        },
        { "thread:1=socket:0.pu:0"
        },

        { "thread:1=numanode:0"
        },
        { "thread:1=numanode:0.core:0"
        },
        { "thread:1=numanode:0.core:0.pu:0"
        },
        { "thread:1=numanode:0.pu:0"
        },

        { "thread:1=core:0"
        },
        { "thread:1=core:0.pu:0"
        },

        { "thread:1=pu:0"
        },

        { "thread:2=core:all.pu:0"
        },
        { "t:0-3=c:0-3.p:0"
        },
        { "" }
    };

    void good()
    {
        for (data_good* t = data; !t->option_.empty(); ++t)
        {
            std::vector<hpx::threads::mask_type> affinities;
            HPX_TEST(hpx::threads::parse_affinity_options(t->option_, affinities));
        }
    }

    char const* const data_bad[] =
    {
        "thread:0=pu:0.socket:0",     // wrong sequence
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

