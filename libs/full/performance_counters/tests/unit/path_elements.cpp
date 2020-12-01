//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/util.hpp>
#include <hpx/modules/testing.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace test {
    ///////////////////////////////////////////////////////////////////////////
    struct data_good
    {
        std::string fullname_;
        std::string typename_;
        hpx::performance_counters::counter_path_elements path_;
    };

    // clang-format off
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
                "",
                2, 1, -1, false)
        },
        {   "/objectname{parentinstancename#*/instancename#*}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename#*",
                "instancename#*",
                "",
                -1, -1, -1, false)
        },
        {   "/objectname{parentinstancename#2/instancename#1}/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "instancename",
                "",
                2, 1, -1, false)
        },
        {   "/objectname{parentinstancename#*/instancename#*}/countername@parameter",
            "/objectname/countername",
               hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename#*",
                "instancename#*",
                "",
                -1, -1, -1, false)
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
                "",
                -1, -1, -1, true)
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
                "",
                -1, -1, -1, false)
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
                "",
                2, 1, -1, false)
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
                "",
                -1, -1, -1, false)
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
                "",
                2, 1, -1, false)
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
                "",
                -1, -1, -1, false)
        },
        {   "/objectname{parentinstancename/instancename#1}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "instancename",
                "",
                -1, 1, -1, false)
        },
        {   "/objectname{parentinstancename/instancename#1}/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "instancename",
                "",
                -1, 1, -1, false)
        },
        {   "/objectname{parentinstancename/moreparent/instancename#1}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "moreparent",
                "instancename",
                -1, -1, 1, false)
        },
        {   "/objectname{parentinstancename/moreparent/instancename#*}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "moreparent",
                "instancename#*",
                -1, -1, -1, false)
        },
        {   "/objectname{parentinstancename/moreparent/instancename#1}"
            "/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "moreparent",
                "instancename",
                -1, -1, 1, false)
        },
        {   "/objectname{parentinstancename/moreparent/instancename#*}"
            "/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "moreparent",
                "instancename#*",
                -1, -1, -1, false)
        },
        {   "/objectname{parentinstancename/moreparent#2/instancename#1}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "moreparent",
                "instancename",
                -1, 2, 1, false)
        },
        {   "/objectname{parentinstancename/moreparent#*/instancename#*}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "moreparent#*",
                "instancename#*",
                -1, -1, -1, false)
        },
        {   "/objectname{parentinstancename/moreparent#2/instancename#1}"
            "/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "moreparent",
                "instancename",
                -1, 2, 1, false)
        },
        {   "/objectname{parentinstancename/moreparent#*/instancename#*}"
            "/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "moreparent#*",
                "instancename#*",
                -1, -1, -1, false)
        },
        {   "/objectname{parentinstancename/instancename}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "instancename",
                "",
                -1, -1, -1, false)
        },
        {   "/objectname{parentinstancename/instancename}/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "instancename",
                "",
                -1, -1, -1, false)
        },
        {   "/objectname{parentinstancename/moreparent/instancename}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "moreparent",
                "instancename",
                -1, -1, -1, false)
        },
        {   "/objectname{parentinstancename/moreparent/instancename}"
            "/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "moreparent",
                "instancename",
                -1, -1, -1, false)
        },
        {   "/objectname{parentinstancename#1}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "",
                "",
                1, -1, -1, false)
        },
        {   "/objectname{parentinstancename#1}/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "",
                "",
                1, -1, -1, false)
        },
        {   "/objectname{parentinstancename}/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "parentinstancename",
                "",
                "",
                -1, -1, -1, false)
        },
        {   "/objectname{parentinstancename}/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "parentinstancename",
                "",
                "",
                -1, -1, -1, false)
        },
        {   "/objectname/countername",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "",
                "",
                "",
                "",
                -1, -1, -1, false)
        },
        {   "/objectname/countername@parameter",
            "/objectname/countername",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "countername",
                "parameter",
                "",
                "",
                "",
                -1, -1, -1, false)
        },
        {   "/objectname",
            "/objectname",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "",
                "",
                "",
                "",
                "",
                -1, -1, -1, false)
        },
        {   "/objectname@parameter",
            "/objectname",
            hpx::performance_counters::counter_path_elements(
                "objectname",
                "",
                "parameter",
                "",
                "",
                "",
                -1, -1, -1, false)
        },
        {   "", "", hpx::performance_counters::counter_path_elements() }
    };
    // clang-format on

    void good()
    {
        hpx::error_code ec;
        for (data_good* t = data; !t->fullname_.empty(); ++t)
        {
            using namespace hpx::performance_counters;

            std::string fullname;
            HPX_TEST_EQ(
                status_valid_data, get_counter_name(t->path_, fullname, ec));
            HPX_TEST_EQ(ec.value(), hpx::success);
            HPX_TEST_EQ(fullname, t->fullname_);

            std::string type_name;
            HPX_TEST(status_valid_data ==
                get_counter_type_name(t->path_, type_name, ec));
            HPX_TEST_EQ(ec.value(), hpx::success);
            HPX_TEST_EQ(type_name, t->typename_);

            counter_path_elements p;

            HPX_TEST(status_valid_data ==
                get_counter_path_elements(t->fullname_, p, ec));
            HPX_TEST_EQ(ec.value(), hpx::success);
            HPX_TEST_EQ(p.objectname_, t->path_.objectname_);
            HPX_TEST_EQ(p.parentinstancename_, t->path_.parentinstancename_);
            HPX_TEST_EQ(p.instancename_, t->path_.instancename_);
            HPX_TEST_EQ(p.subinstancename_, t->path_.subinstancename_);
            HPX_TEST_EQ(p.instanceindex_, t->path_.instanceindex_);
            HPX_TEST_EQ(p.subinstanceindex_, t->path_.subinstanceindex_);
            HPX_TEST_EQ(p.countername_, t->path_.countername_);

            fullname.erase();
            HPX_TEST_EQ(status_valid_data, get_counter_name(p, fullname, ec));
            HPX_TEST_EQ(ec.value(), hpx::success);
            HPX_TEST_EQ(fullname, t->fullname_);

            counter_type_path_elements tp1, tp2;

            HPX_TEST(status_valid_data ==
                get_counter_type_path_elements(t->fullname_, tp1, ec));
            HPX_TEST_EQ(ec.value(), hpx::success);
            HPX_TEST_EQ(tp1.objectname_, t->path_.objectname_);
            HPX_TEST_EQ(tp1.countername_, t->path_.countername_);

            type_name.erase();
            HPX_TEST_EQ(
                status_valid_data, get_counter_type_name(tp1, type_name, ec));
            HPX_TEST_EQ(ec.value(), hpx::success);
            HPX_TEST_EQ(type_name, t->typename_);

            type_name.erase();
            HPX_TEST(status_valid_data ==
                get_full_counter_type_name(tp1, type_name, ec));
            HPX_TEST_EQ(ec.value(), hpx::success);
            if (t->path_.parameters_.empty())
            {
                HPX_TEST_EQ(type_name, t->typename_);
            }
            else
            {
                HPX_TEST_EQ(
                    type_name, t->typename_ + '@' + t->path_.parameters_);
            }

            HPX_TEST(status_valid_data ==
                get_counter_type_path_elements(t->typename_, tp2, ec));
            HPX_TEST_EQ(ec.value(), hpx::success);
            HPX_TEST_EQ(tp2.objectname_, t->path_.objectname_);
            HPX_TEST_EQ(tp2.countername_, t->path_.countername_);

            type_name.erase();
            HPX_TEST_EQ(
                status_valid_data, get_counter_type_name(tp2, type_name, ec));
            HPX_TEST_EQ(ec.value(), hpx::success);
            HPX_TEST_EQ(type_name, t->typename_);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // clang-format off
    char const* const data_bad[] =
    {
        "/{parentinstancename/instancename#1}/countername",
        "/objectname{parentinstancename/instancename#1/countername",
        "/objectname{parentinstancename#/instancename#1}/countername",
        "/objectname{parentinstancename/instancename#/badindex}/countername/badname",
        "/objectname{parentinstancename/instancename/badindex#}/countername/badname",
        "//countername",
        "/objectname{}/countername",
        "/{}/countername",
        "/objectname{/instancename#1}/countername",
        "/objectname{parentinstancename/#1}/countername",
        "/objectname{parentinstancename/instancename#}/countername",
        "/objectname{parentinstancename/instancename#1}/",
        nullptr
    };
    // clang-format on

    void bad()
    {
        using namespace hpx::performance_counters;

        // test non-throwing version
        counter_path_elements p;
        int i = 0;
        for (char const* t = data_bad[0]; nullptr != t; t = data_bad[++i])
        {
            hpx::error_code ec;
            HPX_TEST_EQ(
                status_invalid_data, get_counter_path_elements(t, p, ec));
            HPX_TEST_EQ(ec.value(), hpx::bad_parameter);
        }

        // test throwing version
        i = 0;
        for (char const* t = data_bad[0]; nullptr != t; t = data_bad[++i])
        {
            hpx::error_code ec;
            bool caught_exception = false;
            try
            {
                get_counter_path_elements(t, p);
                HPX_TEST(false);
            }
            catch (hpx::exception const& e)
            {
                HPX_TEST_EQ(e.get_error(), hpx::bad_parameter);
                caught_exception = true;
            }
            HPX_TEST(caught_exception);
        }
    }
}    // namespace test

int hpx_main()
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
    return hpx::init(argc, argv);    // Initialize and run HPX.
}
#endif
