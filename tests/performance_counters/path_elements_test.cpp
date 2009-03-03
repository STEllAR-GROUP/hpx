//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/performance_counters/base_performance_counter.hpp>
#include <boost/detail/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
struct testdata_good
{
    std::string fullname_;
    hpx::performance_counters::counter_path_elements path_;
};

testdata_good data[] = 
{
    {   "/objectname(parentinstancename/instancename#1)/countername",
        {   "objectname",
            "parentinstancename",
            "instancename",
            1,
            "countername"
        }
    },
    {   "/objectname(parentinstancename/moreparent/instancename#1)/countername",
        {   "objectname",
            "parentinstancename/moreparent",
            "instancename",
            1,
            "countername"
        }
    },
    {   "/objectname(parentinstancename/instancename)/countername",
        {   "objectname",
            "parentinstancename",
            "instancename",
            0,
            "countername"
        }
    },
    {   "/objectname(parentinstancename/moreparent/instancename)/countername",
        {   "objectname",
            "parentinstancename/moreparent",
            "instancename",
            0,
            "countername"
        }
    },
    {   "/objectname(instancename#1)/countername",
        {   "objectname",
            "",
            "instancename",
            1,
            "countername"
        }
    },
    {   "/objectname(instancename)/countername",
        {   "objectname",
            "",
            "instancename",
            0,
            "countername"
        }
    },
    {   "/objectname/countername",
        {   "objectname",
            "",
            "",
            0,
            "countername"
        }
    },
    {   "", { "", "", "", 0, "" } }
};

void test_good()
{
    for (testdata_good* t = data; !t->fullname_.empty(); ++t)
    {
        using namespace hpx::performance_counters;

        std::string fullname;
        BOOST_TEST(status_valid_data == get_counter_name(t->path_, fullname));
        BOOST_TEST(fullname == t->fullname_);

        counter_path_elements p;
        p.instanceindex_ = 0;

        BOOST_TEST(status_valid_data == get_counter_path_elements(t->fullname_, p));
        BOOST_TEST(p.objectname_ == t->path_.objectname_);
        BOOST_TEST(p.parentinstancename_ == t->path_.parentinstancename_);
        BOOST_TEST(p.instancename_ == t->path_.instancename_);
        BOOST_TEST(p.instanceindex_ == t->path_.instanceindex_);
        BOOST_TEST(p.countername_ == t->path_.countername_);
    }
}

char const* const testdata_bad[] =
{
    "/(parentinstancename/instancename#1)/countername",
    "/objectname(parentinstancename/instancename#1/countername",
    "/objectname(parentinstancename/instancename#1)/countername/badname",
    "/objectname(parentinstancename/instancename#1/badindex)/countername/badname",
    "//countername",
    "/objectname()/countername",
    "/()/countername",
    "/objectname(/instancename#1)/countername",
    "/objectname(parentinstancename/#1)/countername",
    "/objectname(parentinstancename/instancename#)/countername",
    "/objectname(parentinstancename/instancename#1)/",
    "/objectname",
    "/objectname/countername/badname",
    NULL
};

void test_bad()
{
    using namespace hpx::performance_counters;

    counter_path_elements p;
    int i = 0;
    for (char const* t = testdata_bad[0]; NULL != t; t = testdata_bad[++i])
    {
        BOOST_TEST(status_invalid_data == get_counter_path_elements(t, p));
    }
}

int main()
{
    test_good();
    test_bad();
    return boost::report_errors();
}

