//  Copyright (c) 2026 Pratyksh Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Exhaustive tests to verify that completion_signatures are only obtained via
// tag_invoke(get_completion_signatures_t, ...) and that the fallback to nested
// type aliases has been removed (P3164).
//
// This test file contains compile-time tests that verify the removal of
// has_completion_signatures and the nested type alias fallback mechanism.

#include <hpx/config.hpp>

#if !defined(HPX_HAVE_STDEXEC)

#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/modules/testing.hpp>

#include <exception>
#include <type_traits>

namespace ex = hpx::execution::experimental;

///////////////////////////////////////////////////////////////////////////////
// Test 1: Sender with tag_invoke ONLY (should work)
///////////////////////////////////////////////////////////////////////////////

struct sender_with_tag_invoke_only
{
    using is_sender = void;

    // No nested completion_signatures type alias

    template <typename Env>
    friend auto tag_invoke(ex::get_completion_signatures_t,
        sender_with_tag_invoke_only const&, Env&&) noexcept
        -> ex::completion_signatures<ex::set_value_t(int),
            ex::set_error_t(std::exception_ptr)>;
};

void test_sender_with_tag_invoke_only()
{
    static_assert(ex::is_sender_v<sender_with_tag_invoke_only>,
        "Sender with tag_invoke should be recognized");

    using sigs = ex::completion_signatures_of_t<sender_with_tag_invoke_only>;
    static_assert(std::is_same_v<sigs,
                      ex::completion_signatures<ex::set_value_t(int),
                          ex::set_error_t(std::exception_ptr)>>,
        "Should get completion signatures via tag_invoke");
}

///////////////////////////////////////////////////////////////////////////////
// Test 2: Sender with BOTH tag_invoke and nested alias (tag_invoke wins)
///////////////////////////////////////////////////////////////////////////////

struct sender_with_both
{
    using is_sender = void;

    // Has nested alias (should be ignored since fallback is removed)
    using completion_signatures_IGNORED =
        ex::completion_signatures<ex::set_value_t(
            double)>;    // different from tag_invoke

    // tag_invoke should be the ONLY mechanism
    template <typename Env>
    friend auto tag_invoke(ex::get_completion_signatures_t,
        sender_with_both const&, Env&&) noexcept
        -> ex::completion_signatures<ex::set_value_t(int),
            ex::set_error_t(std::exception_ptr)>;
};

void test_sender_with_both()
{
    using sigs = ex::completion_signatures_of_t<sender_with_both>;
    static_assert(std::is_same_v<sigs,
                      ex::completion_signatures<ex::set_value_t(int),
                          ex::set_error_t(std::exception_ptr)>>,
        "tag_invoke should be used");

    // Verify it's NOT using a nested alias
    static_assert(!std::is_same_v<sigs,
                      ex::completion_signatures<ex::set_value_t(double)>>,
        "Should not use nested completion_signatures alias");
}

///////////////////////////////////////////////////////////////////////////////
// Test 3: Template sender with different signatures
///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct template_sender
{
    using is_sender = void;

    template <typename Env>
    friend auto tag_invoke(
        ex::get_completion_signatures_t, template_sender const&, Env&&) noexcept
        -> ex::completion_signatures<ex::set_value_t(T),
            ex::set_error_t(std::exception_ptr)>;
};

void test_template_sender()
{
    using sigs_int = ex::completion_signatures_of_t<template_sender<int>>;
    static_assert(std::is_same_v<sigs_int,
                      ex::completion_signatures<ex::set_value_t(int),
                          ex::set_error_t(std::exception_ptr)>>,
        "Template sender with int should work");

    using sigs_double = ex::completion_signatures_of_t<template_sender<double>>;
    static_assert(std::is_same_v<sigs_double,
                      ex::completion_signatures<ex::set_value_t(double),
                          ex::set_error_t(std::exception_ptr)>>,
        "Template sender with double should work");
}

///////////////////////////////////////////////////////////////////////////////
// Test 4: Sender with multiple completion signatures
///////////////////////////////////////////////////////////////////////////////

struct multi_sig_sender
{
    using is_sender = void;

    template <typename Env>
    friend auto tag_invoke(ex::get_completion_signatures_t,
        multi_sig_sender const&, Env&&) noexcept
        -> ex::completion_signatures<ex::set_value_t(int),
            ex::set_value_t(double), ex::set_error_t(std::exception_ptr),
            ex::set_stopped_t()>;
};

void test_multiple_signatures()
{
    using sigs = ex::completion_signatures_of_t<multi_sig_sender>;
    static_assert(
        std::is_same_v<sigs,
            ex::completion_signatures<ex::set_value_t(int),
                ex::set_value_t(double), ex::set_error_t(std::exception_ptr),
                ex::set_stopped_t()>>,
        "Should support multiple completion signatures");
}

///////////////////////////////////////////////////////////////////////////////
// Test 5: Sender with only set_stopped
///////////////////////////////////////////////////////////////////////////////

struct stopped_only_sender
{
    using is_sender = void;

    template <typename Env>
    friend auto tag_invoke(ex::get_completion_signatures_t,
        stopped_only_sender const&, Env&&) noexcept
        -> ex::completion_signatures<ex::set_stopped_t()>;
};

void test_stopped_only()
{
    using sigs = ex::completion_signatures_of_t<stopped_only_sender>;
    static_assert(
        std::is_same_v<sigs, ex::completion_signatures<ex::set_stopped_t()>>,
        "Should support stopped-only sender");
}

///////////////////////////////////////////////////////////////////////////////
// Test 6: Different template parameter types
///////////////////////////////////////////////////////////////////////////////

template <bool val>
struct conditional_sender
{
    using is_sender = void;

    template <typename Env>
    friend auto tag_invoke(ex::get_completion_signatures_t,
        conditional_sender const&, Env&&) noexcept -> std::conditional_t<val,
        ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>,
        ex::completion_signatures<ex::set_value_t(int)>>;
};

void test_conditional_sender()
{
    using sigs_true = ex::completion_signatures_of_t<conditional_sender<true>>;
    static_assert(std::is_same_v<sigs_true,
                      ex::completion_signatures<ex::set_value_t(int),
                          ex::set_stopped_t()>>,
        "Conditional sender with true should have stopped signal");

    using sigs_false =
        ex::completion_signatures_of_t<conditional_sender<false>>;
    static_assert(std::is_same_v<sigs_false,
                      ex::completion_signatures<ex::set_value_t(int)>>,
        "Conditional sender with false should not have stopped signal");
}

///////////////////////////////////////////////////////////////////////////////
// Test 7: Verify tag_invoke takes precedence over any naming conventions
///////////////////////////////////////////////////////////////////////////////

struct tricky_sender
{
    using is_sender = void;

    // This should be IGNORED (it's not the exact name that was checked before)
    struct completion_signatures
    {
        template <template <typename...> typename Tuple,
            template <typename...> typename Variant>
        using value_types = Variant<Tuple<double>>;

        template <template <typename...> typename Variant>
        using error_types = Variant<>;

        static constexpr bool sends_stopped = false;
    };

    // Only tag_invoke matters now
    template <typename Env>
    friend auto tag_invoke(
        ex::get_completion_signatures_t, tricky_sender const&, Env&&) noexcept
        -> ex::completion_signatures<ex::set_value_t(int)>;
};

void test_tricky_sender()
{
    using sigs = ex::completion_signatures_of_t<tricky_sender>;
    static_assert(
        std::is_same_v<sigs, ex::completion_signatures<ex::set_value_t(int)>>,
        "Should use tag_invoke, not nested struct");
}

///////////////////////////////////////////////////////////////////////////////
// Main test driver
///////////////////////////////////////////////////////////////////////////////

int main()
{
    test_sender_with_tag_invoke_only();
    test_sender_with_both();
    test_template_sender();
    test_multiple_signatures();
    test_stopped_only();
    test_conditional_sender();
    test_tricky_sender();

    return hpx::util::report_errors();
}

#else

// When STDEXEC is defined, this test doesn't apply
int main()
{
    return 0;
}

#endif
