..
    Copyright (C)      2013 Thomas Heller

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

=============
Testing |hpx|
=============

To ensure correctness of |hpx|, we ship a large variety of unit and regression
tests. The tests are driven by the |ctest|_ tool and are executed automatically 
on each commit to the |hpx| |hpx_github|_ repository. In addition, it is encouraged 
to run the test suite manually to ensure proper operation on your target system. 
If a test fails for your platform, we highly recommend submitting an issue on our 
|hpx_github_issues|_ tracker with detailed information about the target system.

Running tests manually
======================

Running the tests manually is as easy as typing ``make tests && make test``.
This will build all tests and run them once the tests are built successfully.
After the tests have been built, you can invoke separate tests with the help of
the ``ctest`` command. You can list all available test targets using ``make help
| grep tests``. Please see the |ctest_doc|_ for further details.

Issue tracker
=============

If you stumble over a bug or missing feature in |hpx|, please
submit an issue to our |hpx_github_issues|_ page. For more information on how to
submit support requests or other means of getting in contact with the developers,
please see the |support|_ page.

Continuous testing
==================

In addition to manual testing, we run automated tests on various platforms. We also 
run tests on all pull requests using both |circleci|_ and a combination of |cdash|_ 
and |pycicle|_. You can see the dashboards here: |hpx_circleci|_ and |hpx_cdash|_ .
