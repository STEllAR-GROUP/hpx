..
    Copyright (c) 2026 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_supervision:

===========
supervision
===========

The supervision module provides lifecycle event publication, state querying,
and observer registration for actors/components running on local or remote
localities.

Overview
========

Four core operations are exposed, each with a local (synchronous) call and
a remote (locality-qualified, future-returning or ``launch::sync_policy``)
call:

Publishing events
------------------

.. cpp:function:: hpx::future<hpx::supervision::publish_result> hpx::supervision::publish_event(hpx::id_type const& locality, hpx::id_type const& target, hpx::supervision::event ev, std::uint64_t epoch = 0)
.. cpp:function:: hpx::supervision::publish_result hpx::supervision::publish_event(hpx::launch::sync_policy, hpx::id_type const& locality, hpx::id_type const& target, hpx::supervision::event ev, std::uint64_t epoch = 0, hpx::error_code& ec = hpx::throws)
.. cpp:function:: hpx::supervision::publish_result hpx::supervision::publish_event(hpx::id_type const& target, hpx::supervision::event ev, std::uint64_t epoch = 0, hpx::error_code& ec = hpx::throws)

    Publish a lifecycle event for ``target``. Events are visible immediately
    to local observers; remote observers are notified within roughly one to
    two parcel round-trips. Publishing non-terminal events is not idempotent
    - each call creates a distinct, timestamped record. ``event::completed``
    and ``event::failed`` are latched: the first terminal publication for a
    target returns ``publish_result::applied``; every later terminal
    publication for that target is a no-op that returns
    ``publish_result::already_terminal``.

Querying lifecycle state
-------------------------

.. cpp:function:: hpx::future<hpx::supervision::lifecycle_state> hpx::supervision::query_state(hpx::id_type const& locality, hpx::id_type const& target)
.. cpp:function:: hpx::supervision::lifecycle_state hpx::supervision::query_state(hpx::launch::sync_policy, hpx::id_type const& locality, hpx::id_type const& target, hpx::error_code& ec = hpx::throws)
.. cpp:function:: hpx::supervision::lifecycle_state hpx::supervision::query_state(hpx::id_type const& target, hpx::error_code& ec = hpx::throws)

    Query the most recently observed lifecycle state for ``target``. Includes
    a sequence number for gap detection and a staleness error code for remote
    queries whose result may lag the latest event.

Registering observers
----------------------

.. cpp:function:: hpx::future<hpx::id_type> hpx::supervision::register_observer(hpx::id_type const& locality, hpx::id_type const& target, hpx::supervision::lifecycle_callback const& callback, std::optional<std::uint64_t> epoch_filter = std::nullopt)
.. cpp:function:: hpx::id_type hpx::supervision::register_observer(hpx::launch::sync_policy, hpx::id_type const& locality, hpx::id_type const& target, hpx::supervision::lifecycle_callback const& callback, std::optional<std::uint64_t> epoch_filter = std::nullopt, hpx::error_code& ec = hpx::throws)
.. cpp:function:: hpx::id_type hpx::supervision::register_observer(hpx::id_type const& target, hpx::supervision::lifecycle_callback const& callback, std::optional<std::uint64_t> epoch_filter = std::nullopt, hpx::error_code& ec = hpx::throws)

    Register ``callback`` to be invoked on lifecycle events of ``target``,
    returning an observer handle usable with ``unregister_observer``. Local
    callbacks fire synchronously within the publish call; remote callbacks
    fire via a retried parcel. If ``epoch_filter`` is set, the observer only
    receives notifications (including the initial state snapshot delivered
    at registration time) whose epoch matches; by default the observer
    receives notifications for every epoch.

Unregistering observers
------------------------

.. cpp:function:: hpx::future<void> hpx::supervision::unregister_observer(hpx::id_type const& locality, hpx::id_type const& observer_handle)
.. cpp:function:: void hpx::supervision::unregister_observer(hpx::launch::sync_policy, hpx::id_type const& locality, hpx::id_type const& observer_handle, hpx::error_code& ec = hpx::throws)
.. cpp:function:: void hpx::supervision::unregister_observer(hpx::id_type const& observer_handle, hpx::error_code& ec = hpx::throws)

    Unregister a previously registered observer.

Waiting for terminal events
-----------------------------

.. cpp:function:: hpx::future<hpx::supervision::lifecycle_state> hpx::supervision::await_terminal(hpx::id_type const& locality, hpx::id_type const& target, std::uint64_t epoch = 0, std::chrono::steady_clock::duration timeout = std::chrono::steady_clock::duration::max())
.. cpp:function:: hpx::supervision::lifecycle_state hpx::supervision::await_terminal(hpx::launch::sync_policy, hpx::id_type const& locality, hpx::id_type const& target, std::uint64_t epoch = 0, std::chrono::steady_clock::duration timeout = std::chrono::steady_clock::duration::max(), hpx::error_code& ec = hpx::throws)
.. cpp:function:: hpx::future<hpx::supervision::lifecycle_state> hpx::supervision::await_terminal(hpx::id_type const& target, std::uint64_t epoch = 0, std::chrono::steady_clock::duration timeout = std::chrono::steady_clock::duration::max())

    Asynchronously wait, one-shot and without blocking a worker thread, and the
    ``launch::sync_policy`` overload waits synchronously until ``target`` reaches
    a terminal event (``event::completed`` or ``event::failed``) within ``epoch``.
    If a terminal event has already been recorded for
    ``target`` in ``epoch``, the returned future is immediately satisfied
    with that state. Waits are scoped to a single epoch: if ``target``'s
    epoch advances past ``epoch`` before a terminal event occurs, any
   outstanding waiter for the stale epoch is invalidated. A waiter that
    reaches neither of these outcomes is bounded by ``timeout`` (or a
    built-in default if ``timeout`` is left at its sentinel value
    ``std::chrono::steady_clock::duration::max()``), after which it is
    swept and invalidated. Dispatch is routed locally when ``target``'s
    supervision manager lives on the calling locality, and remotely (via a
    registered distributed action) otherwise.

See the :ref:`API reference <modules_supervision_api>` of this module for
more details.
