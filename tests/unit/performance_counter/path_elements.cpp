//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace test
{
    ///////////////////////////////////////////////////////////////////////////
    struct data_good
    {
        std::string fullname_;
        std::string typename_;
        hpx::performance_counters::counter_path_elements path_;
    };

    data_good data[] =
    {
        {   "/objectname{parentinstancename#2/instancename#1}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "instancename",
                2, 1, false)
        },
        {   "/objectname{parentinstancename#*/instancename#*}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename#*",
                "instancename#*",
                -1, -1, false)
        },
        {   "/objectname{parentinstancename#2/instancename#1}/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "instancename",
                2, 1, false)
        },
        {   "/objectname{parentinstancename#*/instancename#*}/countername@parameter",
            "/objectname/countername",
               hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename#*",
                "instancename#*",
                -1, -1, false)
        },
        {   "/objectname{/objectname{parentinstancename#2/instancename#1}"
            "/countername}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "/objectname{parentinstancename#2/instancename#1}/countername",
                "",
                -1, -1, true)
        },
        {   "/objectname{/objectname{parentinstancename#2/instancename#1}"
            "/countername}/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "/objectname{parentinstancename#2/instancename#1}/countername",
                "",
                -1, -1, false)
        },
        {   "/objectname{parentinstancename#2/instancename#1}"
            "/countername/morecountername",
            "/objectname/countername/morecountername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername/morecountername",
                "",
                "parentinstancename",
                "instancename",
                2, 1, false)
        },
        {   "/objectname{parentinstancename#*/instancename#*}"
            "/countername/morecountername",
            "/objectname/countername/morecountername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername/morecountername",
                "",
                "parentinstancename#*",
                "instancename#*",
                -1, -1, false)
        },
        {   "/objectname{parentinstancename#2/instancename#1}"
            "/countername/morecountername@parameter",
            "/objectname/countername/morecountername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername/morecountername",
                "parameter",
                "parentinstancename",
                "instancename",
                2, 1, false)
        },
        {   "/objectname{parentinstancename#*/instancename#*}"
            "/countername/morecountername@parameter",
            "/objectname/countername/morecountername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername/morecountername",
                "parameter",
                "parentinstancename#*",
                "instancename#*",
                -1, -1, false)
        },
        {   "/objectname{parentinstancename/instancename#1}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "instancename",
                -1, 1, false)
        },
        {   "/objectname{parentinstancename/instancename#1}/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "instancename",
                -1, 1, false)
        },
        {   "/objectname{parentinstancename/moreparent/instancename#1}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "moreparent/instancename",
                -1, 1, false)
        },
        {   "/objectname{parentinstancename/moreparent/instancename#*}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "moreparent/instancename#*",
                -1, -1, false)
        },
        {   "/objectname{parentinstancename/moreparent/instancename#1}"
            "/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "moreparent/instancename",
                -1, 1, false)
        },
        {   "/objectname{parentinstancename/moreparent/instancename#*}"
            "/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "moreparent/instancename#*",
                -1, -1, false)
        },
        {   "/objectname{parentinstancename/instancename}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "instancename",
                -1, -1, false)
        },
        {   "/objectname{parentinstancename/instancename}"
            "/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "instancename",
                -1, -1, false)
        },
        {   "/objectname{parentinstancename/moreparent"
            "/instancename}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "moreparent/instancename",
                -1, -1, false)
        },
        {   "/objectname{parentinstancename/moreparent/instancename}"
            "/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "moreparent/instancename",
                -1, -1, false)
        },
        {   "/objectname{parentinstancename#1}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "",
                1, -1, false)
        },
        {   "/objectname{parentinstancename#1}/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "",
                1, -1, false)
        },
        {   "/objectname{parentinstancename}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "",
                -1, -1, false)
        },
        {   "/objectname{parentinstancename}/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "",
                -1, -1, false)
        },
        {   "/objectname/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "",
                "",
                -1, -1, false)
        },
        {   "/objectname/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "",
                "",
                -1, -1, false)
        },
        {   "/objectname",
            "/objectname",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "",
                "",
                "",
                "",
                -1, -1, false)
        },
        {   "/objectname@parameter",
            "/objectname",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "",
                "parameter",
                "",
                "",
                -1, -1, false)
        },
        {   "", "", hpx::performance_counters::counter_path_elements() }
    };

    void good()
    {
        hpx::error_code ec;
        for (data_good* t = data; !t->fullname_.empty(); ++t)
        {
            using namespace hpx::performance_counters;

            std::string fullname;
            HPX_TEST(status_valid_data == get_counter_name(t->path_, fullname, ec));
            HPX_TEST(ec.value() == hpx::success);
            HPX_TEST(fullname == t->fullname_);

            std::string type_name;
            HPX_TEST(status_valid_data ==
                get_counter_type_name(t->path_, type_name, ec));
            HPX_TEST(ec.value() == hpx::success);
            HPX_TEST(type_name == t->typename_);

            counter_path_elements p;

            HPX_TEST(status_valid_data ==
                get_counter_path_elements(t->fullname_, p, ec));
            HPX_TEST(ec.value() == hpx::success);
            HPX_TEST(p.objectname_ == t->path_.objectname_);
            HPX_TEST(p.parentinstancename_ == t->path_.parentinstancename_);
            HPX_TEST(p.instancename_ == t->path_.instancename_);
            HPX_TEST(p.instanceindex_ == t->path_.instanceindex_);
            HPX_TEST(p.countername_ == t->path_.countername_);

            fullname.erase();
            HPX_TEST(status_valid_data == get_counter_name(p, fullname, ec));
            HPX_TEST(ec.value() == hpx::success);
            HPX_TEST(fullname == t->fullname_);

            counter_type_path_elements tp1, tp2;

            HPX_TEST(status_valid_data ==
                get_counter_type_path_elements(t->fullname_, tp1, ec));
            HPX_TEST(ec.value() == hpx::success);
            HPX_TEST(tp1.objectname_ == t->path_.objectname_);
            HPX_TEST(tp1.countername_ == t->path_.countername_);

            type_name.erase();
            HPX_TEST(status_valid_data == get_counter_type_name(tp1, type_name, ec));
            HPX_TEST(ec.value() == hpx::success);
            HPX_TEST(type_name == t->typename_);

            type_name.erase();
            HPX_TEST(status_valid_data ==
                get_full_counter_type_name(tp1, type_name, ec));
            HPX_TEST(ec.value() == hpx::success);
            if (t->path_.parameters_.empty()) {
                HPX_TEST(type_name == t->typename_);
            }
            else {
                HPX_TEST(type_name == t->typename_ + '@' + t->path_.parameters_);
            }

            HPX_TEST(status_valid_data ==
                get_counter_type_path_elements(t->typename_, tp2, ec));
            HPX_TEST(ec.value() == hpx::success);
            HPX_TEST(tp2.objectname_ == t->path_.objectname_);
            HPX_TEST(tp2.countername_ == t->path_.countername_);

            type_name.erase();
            HPX_TEST(status_valid_data == get_counter_type_name(tp2, type_name, ec));
            HPX_TEST(ec.value() == hpx::success);
            HPX_TEST(type_name == t->typename_);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    char const* const data_bad[] =
    {
        "/{parentinstancename/instancename#1}/countername",
        "/objectname{parentinstancename/instancename#1/countername",
        "/objectname{parentinstancename#/instancename#1}/countername",
        "/objectname{parentinstancename/instancename#1/badindex}/countername/badname",
        "//countername",
        "/objectname{}/countername",
        "/{}/countername",
        "/objectname{/instancename#1}/countername",
        "/objectname{parentinstancename/#1}/countername",
        "/objectname{parentinstancename/instancename#}/countername",
        "/objectname{parentinstancename/instancename#1}/",
        NULL
    };

    void bad()
    {
        using namespace hpx::performance_counters;

        // test non-throwing version
        counter_path_elements p;
        int i = 0;
        for (char const* t = data_bad[0]; NULL != t; t = data_bad[++i])
        {
            hpx::error_code ec;
            HPX_TEST_EQ(status_invalid_data, get_counter_path_elements(t, p, ec));
            HPX_TEST_EQ(ec.value(), hpx::bad_parameter);
        }

        // test throwing version
        i = 0;
        for (char const* t = data_bad[0]; NULL != t; t = data_bad[++i])
        {
            hpx::error_code ec;
            bool caught_exception = false;
            try {
                get_counter_path_elements(t, p);
                HPX_TEST(false);
            }
            catch (hpx::exception const& e) {
                HPX_TEST_EQ(e.get_error(), hpx::bad_parameter);
                caught_exception = true;
            }
            HPX_TEST(caught_exception);
        }
    }
}

int hpx_main(boost::program_options::variables_map& vm)
{
    {
        test::good();
        test::bad();
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init(HPX_APPLICATION_STRING, argc, argv);   // Initialize and run HPX.
}

