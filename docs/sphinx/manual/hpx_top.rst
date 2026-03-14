..
    Copyright (c) 2026 Arpit Khandelwal

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_top:

=======
HPX-Top
=======

|hpx-top| is a real-time, terminal-based dashboard designed for monitoring the
health and performance of distributed |hpx| applications. Inspired by tools like
``htop`` and ``btop``, it provides a high-level overview of system utilization,
network flow, and internal runtime metrics.

Features
========

* **Locality-Aware Monitoring**: Automatically detects and displays metrics for
  all localities in a running distributed application.
* **Thread Pool Utilization**: Visualizes real-time usage of worker threads with
  instantaneous counts and historical sparkline graphs.
* **Parcel Throughput**: Tracks incoming and outgoing parcels, helping to identify
  network-bound bottlenecks or communication hotspots.
* **AGAS Health Monitor**: Displays cache hit and miss rates for the Application
  Global Address Space (AGAS), crucial for diagnosing naming service bottlenecks.

Usage
=====

|hpx-top| is implemented as a Python script that acts as a wrapper around your |hpx|
application. It automatically configures the necessary performance counters and
output formats.

Prerequisites
-------------

The tool requires the ``rich`` Python library for rendering the terminal UI:

.. code-block:: shell-session

   $ pip install rich

Running HPX-Top
---------------

To monitor an |hpx| application, launch it through ``hpx-top.py``:

.. code-block:: shell-session

   $ python3 tools/hpx-top.py ./your_application --hpx:threads=4 [other HPX flags]

Mock Mode
---------

You can also run |hpx-top| in mock mode to explore the interface without a running
|hpx| application:

.. code-block:: shell-session

   $ python3 tools/hpx-top.py --mock

Technical Details
=================

The tool leverages |hpx|'s built-in performance counter framework. It launches the
target application with specific flags (``--hpx:print-counter-interval`` and
``--hpx:print-counter-format=csv-short``) and parses the standard output in a
non-blocking background thread to update the TUI.
