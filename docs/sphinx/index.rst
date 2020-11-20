..
    Copyright (C) 2007-2015 Hartmut Kaiser
    Copyright (C) 2016-2018 Adrian Serio

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

===================================
Welcome to the |hpx| documentation!
===================================

If you're new to |hpx| you can get started with the :ref:`quickstart` guide.
Don't forget to read the :ref:`terminology` section to learn about the most
important concepts in |hpx|. The :ref:`examples` give you a feel for how it is
to write real |hpx| applications and the :ref:`manual` contains detailed
information about everything from building |hpx| to debugging it. There are
links to blog posts and videos about |hpx| in :ref:`additional_material`.

If you can't find what you're looking for in the documentation, please:

* open an issue on `GitHub <hpx_github_issues_>`_;
* contact us on `IRC <stellar_irc_>`_, the HPX channel on the `C++ Slack
  <cpplang_slack_>`_, or on our `mailing list <stellar_list_>`_; or
* read or ask questions tagged with |hpx| on `StackOverflow
  <hpx_stackoverflow_>`_.

See :ref:`citing_hpx` for details on how to cite |hpx| in publications. See
:ref:`hpx_users` for a list of institutions and projects using |hpx|.

What is |hpx|?
==============

|hpx| is a C++ Standard Library for Concurrency and Parallelism. It implements
all of the corresponding facilities as defined by the C++ Standard.
Additionally, in |hpx| we implement functionalities proposed as part of the
ongoing C++ standardization process. We also extend the C++ Standard APIs to the
distributed case. |hpx| is developed by the |stellar| group (see :ref:`people`).

The goal of |hpx| is to create a high quality, freely available, open source
implementation of a new programming model for conventional systems, such as
classic Linux based Beowulf clusters or multi-socket highly parallel SMP nodes.
At the same time, we want to have a very modular and well designed runtime
system architecture which would allow us to port our implementation onto new
computer system architectures. We want to use real-world applications to drive
the development of the runtime system, coining out required functionalities and
converging onto a stable API which will provide a smooth migration path for
developers.

The API exposed by |hpx| is not only modeled after the interfaces defined by the
C++11/14/17/20 ISO standard. It also adheres to the programming guidelines used
by the Boost collection of C++ libraries. We aim to improve the scalability of
today's applications and to expose new levels of parallelism which are necessary
to take advantage of the exascale systems of the future.

What's so special about |hpx|?
==============================

* HPX exposes a uniform, standards-oriented API for ease of programming parallel
  and distributed applications.
* It enables programmers to write fully asynchronous code using hundreds of
  millions of threads.
* HPX provides unified syntax and semantics for local and remote operations.
* HPX makes concurrency manageable with dataflow and future based
  synchronization.
* It implements a rich set of runtime services supporting a broad range of use
  cases.
* HPX exposes a uniform, flexible, and extendable performance counter framework
  which can enable runtime adaptivity
* It is designed to solve problems conventionally considered to be
  scaling-impaired.
* HPX has been designed and developed for systems of any scale, from
  hand-held devices to very large scale systems.
* It is the first fully functional implementation of the ParalleX execution
  model.
* HPX is published under a liberal open-source license and has an open, active,
  and thriving developer community.

.. toctree::
   :caption: User documentation
   :maxdepth: 2

   why_hpx
   quickstart
   terminology
   examples
   manual
   additional_material

.. include:: libs/index.rst

.. toctree::
   :caption: Reference
   :maxdepth: 2

   api

.. toctree::
   :caption: Developer documentation
   :maxdepth: 2

   contributing

.. toctree::
   :caption: Other
   :maxdepth: 2

   releases
   citing
   users
   about_hpx


Index
=====

* :ref:`genindex`

