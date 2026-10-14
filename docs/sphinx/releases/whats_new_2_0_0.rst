..
    Copyright (C) 2007-2025 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_2_0_0:

============================
|hpx| V2.0.0 (TBD)
============================

General changes
===============

- **Fixed-size SIMD policies**: added ``hpx::execution::fixed_size_simd<N>``
  and ``hpx::execution::par_fixed_size_simd<N>`` (plus their ``task``
  variants), which select the number of vector lanes explicitly. Execution
  policy properties, including ``num_lanes``, are exposed through
  ``hpx::execution::policy_traits``.
- Fixed ``hpx::any_of`` with an execution policy to return ``false`` for an
  empty range.

Breaking changes
================

- ``hpx::experimental::for_loop_n_strided`` now invokes the loop body exactly
  ``n`` times, as required by N4755, 7.2.4, paragraph 2.1. It used to read
  ``n`` as the distance to cover and invoked the body ``ceil(n / stride)``
  times instead, so code that scaled ``n`` by the stride to work around this
  has to stop doing that.
- Negative strides are no longer supported by
  ``hpx::experimental::for_loop_strided``,
  ``hpx::experimental::for_loop_n_strided`` and
  ``hpx::ranges::experimental::for_loop_strided``. N4755, 7.2.4,
  paragraph 2.1 only describes a forward traversal, so the stride is now
  required to be positive.
- A live-out induction variable handed to one of the strided ``for_loop``
  algorithms is now advanced by the number of iterations instead of by the
  distance those iterations cover. The value it ended up with used to be too
  large by a factor of the stride.
- The execution-policy overloads of ``hpx::ranges::all_of``, ``any_of``,
  ``none_of``, ``count``, ``count_if``, ``contains``, ``equal``,
  ``starts_with``, ``ends_with``, ``contains_subrange``, ``is_partitioned``,
  and ``lexicographical_compare`` now follow P3179. Range arguments must model
  ``std::ranges::random_access_range`` and ``std::ranges::sized_range``;
  iterator/sentinel arguments must model ``std::random_access_iterator`` and
  ``std::sized_sentinel_for``. Overloads without an execution policy retain
  their existing constraints. In particular, the ``contains``
  iterator/sentinel overload requires both types to model
  ``std::input_iterator``.
- The P3179-aligned overloads use standard ranges callable defaults and pass
  predicates and projections by value. Their concept-based constraints may
  produce different diagnostics than the previous iterator ``static_assert``
  checks. Code that explicitly names HPX's former default callable types may
  need to use the corresponding ``std::ranges`` callable types.
- The ``count`` and ``contains`` task-policy overloads store an owned copy of
  copyable search values in their asynchronous operation state. Noncopyable
  lvalues, including arrays, are borrowed and must remain valid until the
  returned future becomes ready, even if the future is discarded.

Closed issues
=============

Closed pull requests
====================
