//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/performance_counters/counters.hpp>
#include <boost/detail/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
struct testdata_good
{
    std::string fullname_;
    std::string typename_;
    hpx::performance_counters::counter_path_elements path_;
};

testdata_good data[] = 
{
    {   "/objectname(parentinstancename/instancename#1)/countername",
        "/objectname/countername",
        hpx::performance_counters::counter_path_elements(
            "objectname",
            "countername",
            "parentinstancename",
            "instancename",
            1
        )
    },
    {   "/objectname(parentinstancename/moreparent/instancename#1)/countername",
        "/objectname/countername",
        hpx::performance_counters::counter_path_elements(
            "objectname",
            "countername",
            "parentinstancename/moreparent",
            "instancename",
            1
        )
    },
    {   "/objectname(parentinstancename/instancename)/countername",
        "/objectname/countername",
        hpx::performance_counters::counter_path_elements(
            "objectname",
            "countername",
            "parentinstancename",
            "instancename",
            0
        )
    },
    {   "/objectname(parentinstancename/moreparent/instancename)/countername",
        "/objectname/countername",
        hpx::performance_counters::counter_path_elements(
            "objectname",
            "countername",
            "parentinstancename/moreparent",
            "instancename",
            0
        )
    },
    {   "/objectname(instancename#1)/countername",
        "/objectname/countername",
        hpx::performance_counters::counter_path_elements(
            "objectname",
            "countername",
            "",
            "instancename",
            1
        )
    },
    {   "/objectname(instancename)/countername",
        "/objectname/countername",
        hpx::performance_counters::counter_path_elements(
            "objectname",
            "countername",
            "",
            "instancename",
            0
        )
    },
    {   "/objectname/countername",
        "/objectname/countername",
        hpx::performance_counters::counter_path_elements(
            "objectname",
            "countername",
            "",
            "",
            0
        )
    },
    {   "", "", hpx::performance_counters::counter_path_elements() }
};

void test_good()
{
    hpx::error_code ec;
    for (testdata_good* t = data; !t->fullname_.empty(); ++t)
    {
        using namespace hpx::performance_counters;

        std::string fullname;
        BOOST_TEST(status_valid_data == get_counter_name(t->path_, fullname, ec));
        BOOST_TEST(ec.value() == hpx::success);
        BOOST_TEST(fullname == t->fullname_);

        std::string type_name;
        BOOST_TEST(status_valid_data == get_counter_name(
            (counter_type_path_elements const&)t->path_, type_name, ec));
        BOOST_TEST(ec.value() == hpx::success);
        BOOST_TEST(type_name == t->typename_);

        counter_path_elements p;
        p.instanceindex_ = 0;

        BOOST_TEST(status_valid_data == get_counter_path_elements(t->fullname_, p, ec));
        BOOST_TEST(ec.value() == hpx::success);
        BOOST_TEST(p.objectname_ == t->path_.objectname_);
        BOOST_TEST(p.parentinstancename_ == t->path_.parentinstancename_);
        BOOST_TEST(p.instancename_ == t->path_.instancename_);
        BOOST_TEST(p.instanceindex_ == t->path_.instanceindex_);
        BOOST_TEST(p.countername_ == t->path_.countername_);

        counter_type_path_elements tp1, tp2;

        BOOST_TEST(status_valid_data == get_counter_path_elements(t->fullname_, tp1, ec));
        BOOST_TEST(ec.value() == hpx::success);
        BOOST_TEST(tp1.objectname_ == t->path_.objectname_);
        BOOST_TEST(tp1.countername_ == t->path_.countername_);

        BOOST_TEST(status_valid_data == get_counter_path_elements(t->typename_, tp2, ec));
        BOOST_TEST(ec.value() == hpx::success);
        BOOST_TEST(tp2.objectname_ == t->path_.objectname_);
        BOOST_TEST(tp2.countername_ == t->path_.countername_);
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

    // test non-throwing version
    hpx::error_code ec;
    counter_path_elements p;
    int i = 0;
    for (char const* t = testdata_bad[0]; NULL != t; t = testdata_bad[++i])
    {
        BOOST_TEST(status_invalid_data == get_counter_path_elements(t, p, ec));
        BOOST_TEST(ec.value() == hpx::bad_parameter);
    }

    // test throwing version
    i = 0;
    for (char const* t = testdata_bad[0]; NULL != t; t = testdata_bad[++i])
    {
        try {
            get_counter_path_elements(t, p);
            BOOST_TEST(false);
        }
        catch (hpx::exception const& e) {
            BOOST_TEST(e.get_error() == hpx::bad_parameter);
        }
    }
}

int main()
{
    test_good();
    test_bad();
    return boost::report_errors();
}

