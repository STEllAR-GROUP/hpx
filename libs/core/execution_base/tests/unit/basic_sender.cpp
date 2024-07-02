//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

static std::size_t friend_tag_invoke_connect_calls = 0;
static std::size_t tag_invoke_connect_calls = 0;

struct non_sender_1
{
};

struct immovable
{
    immovable() = default;
    immovable(immovable&&) = delete;
    immovable& operator=(immovable&&) = delete;
};

struct non_sender_2
{
    struct completion_signatures
    {
        template <template <class...> class Variant>
        using error_types = Variant<>;

        static constexpr bool sends_stopped = false;
    };
};

struct non_sender_3
{
    struct completion_signatures
    {
        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<>>;

        static constexpr bool sends_stopped = false;
    };
};

struct non_sender_4
{
    struct completion_signatures
    {
        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<>>;

        template <template <class...> class Variant>
        using error_types = Variant<>;
    };
};

struct non_sender_5
{
    struct completion_signatures
    {
        static constexpr bool sends_stopped = false;
    };
};

struct non_sender_6
{
    struct completion_signatures
    {
        template <template <class...> class Variant>
        using error_types = Variant<>;
    };
};

struct non_sender_7
{
    struct completion_signatures
    {
        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<>>;
    };
};

struct example_receiver
{
#ifdef HPX_HAVE_STDEXEC
    using receiver_concept = ex::receiver_t;
#else
    using is_receiver = void;
#endif
    friend void tag_invoke(
        ex::set_error_t, example_receiver&&, std::exception_ptr) noexcept
    {
    }
    friend void tag_invoke(ex::set_stopped_t, example_receiver&&) noexcept {}
    friend void tag_invoke(ex::set_value_t, example_receiver&& r, int v)
#ifdef HPX_HAVE_STDEXEC
        noexcept
#endif
    {
        r.i = v;
    }

    int i = -1;
};

template <typename... T>
struct receiver_2
{
#ifdef HPX_HAVE_STDEXEC
    using receiver_concept = ex::receiver_t;
#else
    using is_receiver = void;
#endif
    friend void tag_invoke(
        ex::set_error_t, receiver_2&&, std::exception_ptr) noexcept
    {
    }
    friend void tag_invoke(ex::set_stopped_t, receiver_2&&) noexcept {}
    friend void tag_invoke(ex::set_value_t, receiver_2&&, T...)
#ifdef HPX_HAVE_STDEXEC
        noexcept
#endif
    {
    }
};

struct sender_1
{
    using is_sender = void;
#ifdef HPX_HAVE_STDEXEC
    using completion_signatures =
        ex::completion_signatures<ex::set_value_t(int),
            ex::set_error_t(std::exception_ptr)>;
#else
    struct completion_signatures
    {
        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<int>>;

        template <template <class...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_stopped = false;
    };
#endif

    struct operation_state : immovable
    {
        example_receiver& r;
        friend void tag_invoke(ex::start_t, operation_state& os) noexcept
        {
            ex::set_value(std::move(os.r), 4711);
        };
    };

    friend operation_state tag_invoke(
        ex::connect_t, sender_1&&, example_receiver& r)
    {
        ++friend_tag_invoke_connect_calls;
        return {{}, r};
    }
};

struct sender_2
{
    using is_sender = void;
#ifdef HPX_HAVE_STDEXEC
    using completion_signatures =
        ex::completion_signatures<ex::set_value_t(int),
            ex::set_error_t(std::exception_ptr)>;
#else
    struct completion_signatures
    {
        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<int>>;

        template <template <class...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_stopped = false;
    };
#endif

    struct operation_state : immovable
    {
        example_receiver& r;
        friend void tag_invoke(ex::start_t, operation_state& os) noexcept
        {
            ex::set_value(std::move(os.r), 4711);
        };
    };
};

sender_2::operation_state tag_invoke(
    ex::connect_t, sender_2, example_receiver& r)
{
    ++tag_invoke_connect_calls;
    return {{}, r};
}

struct sender_3
{
    using is_sender = void;

    using completion_signatures =
        ex::completion_signatures<ex::set_value_t(int),
            ex::set_error_t(std::exception_ptr)>;

    struct operation_state : immovable
    {
        example_receiver& r;
        friend void tag_invoke(ex::start_t, operation_state& os) noexcept
        {
            ex::set_value(std::move(os.r), 4711);
        };
    };

    friend operation_state tag_invoke(
        ex::connect_t, sender_3&&, example_receiver& r)
    {
        ++friend_tag_invoke_connect_calls;
        return {{}, r};
    }
};

template <bool val, typename T>
struct sender_4
{
    using is_sender = void;
#ifdef HPX_HAVE_STDEXEC
    using completion_signatures = std::conditional_t<val,
        ex::completion_signatures<ex::set_value_t(T),
            ex::set_error_t(std::exception_ptr), ex::set_stopped_t()>,
        ex::completion_signatures<ex::set_value_t(T),
            ex::set_error_t(std::exception_ptr)>>;
#else
    struct completion_signatures
    {
        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<T>>;

        template <template <class...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_stopped = val;
    };
#endif
};

static std::size_t void_receiver_set_value_calls = 0;

struct void_receiver
{
    friend void tag_invoke(
        ex::set_error_t, void_receiver&&, std::exception_ptr) noexcept
    {
    }
    friend void tag_invoke(ex::set_stopped_t, void_receiver&&) noexcept {}
    friend void tag_invoke(ex::set_value_t, void_receiver&&)
#ifdef HPX_HAVE_STDEXEC
        noexcept
#endif
    {
        ++void_receiver_set_value_calls;
    }
};

int main()
{
#ifdef HPX_HAVE_STDEXEC
    // different requirements
#else
    using ex::detail::has_sender_types_v;
    static_assert(!has_sender_types_v<void>,
        "void should not have completion_signatures");
    static_assert(!has_sender_types_v<std::nullptr_t>,
        "std::nullptr_t should not have completion_signatures");
    static_assert(!has_sender_types_v<int>,
        "non_sender_1 should not have completion_signatures");
    static_assert(!has_sender_types_v<double>,
        "non_sender_1 should not have completion_signatures");
    static_assert(!has_sender_types_v<non_sender_1>,
        "non_sender_1 should not have completion_signatures");
    static_assert(!has_sender_types_v<non_sender_2>,
        "non_sender_2 should not have completion_signatures");
    static_assert(!has_sender_types_v<non_sender_3>,
        "non_sender_3 should not have completion_signatures");
    static_assert(!has_sender_types_v<non_sender_4>,
        "non_sender_4 should not have completion_signatures");
    static_assert(!has_sender_types_v<non_sender_5>,
        "non_sender_5 should not have completion_signatures");
    static_assert(!has_sender_types_v<non_sender_6>,
        "non_sender_6 should not have completion_signatures");
    static_assert(!has_sender_types_v<non_sender_7>,
        "non_sender_7 should not have completion_signatures");
#endif

    using ex::is_sender_to_v;
    using ex::is_sender_v;

    static_assert(!is_sender_v<void>, "void is not a sender");
    static_assert(
        !is_sender_v<std::nullptr_t>, "std::nullptr_t is not a sender");
    static_assert(!is_sender_v<int>, "int is not a sender");
    static_assert(!is_sender_v<double>, "double is not a sender");
    static_assert(!is_sender_v<non_sender_1>, "non_sender_1 is not a sender");
    static_assert(!is_sender_v<non_sender_2>, "non_sender_2 is not a sender");
    static_assert(!is_sender_v<non_sender_3>, "non_sender_3 is not a sender");
    static_assert(!is_sender_v<non_sender_4>, "non_sender_4 is not a sender");
    static_assert(!is_sender_v<non_sender_5>, "non_sender_5 is not a sender");
    static_assert(!is_sender_v<non_sender_6>, "non_sender_6 is not a sender");
    static_assert(!is_sender_v<non_sender_7>, "non_sender_7 is not a sender");
    static_assert(is_sender_v<sender_1>, "sender_1 is a sender");
    static_assert(is_sender_v<sender_2>, "sender_2 is a sender");
    static_assert(is_sender_v<sender_3>, "sender_3 is a sender");
    static_assert(is_sender_v<sender_4<true, int>>, "sender_4 is a sender");
    static_assert(is_sender_v<sender_4<false, int>>, "sender_4 is a sender");

#ifdef HPX_HAVE_STDEXEC
    // we need to be more specific now
    static_assert(ex::sender_to<sender_1, example_receiver&>,
        "sender_1 is a sender to example_receiver");
#else
    static_assert(ex::is_sender_to_v<sender_1, example_receiver>,
        "sender_1 is a sender to example_receiver");
#endif
    static_assert(!is_sender_to_v<sender_1, non_sender_1>,
        "sender_1 is not a sender to non_sender_1");
    static_assert(!is_sender_to_v<sender_1, sender_1>,
        "sender_1 is not a sender to sender_1");
    static_assert(is_sender_to_v<sender_2, example_receiver&>,
        "sender_2 is a sender to example_receiver");
    static_assert(!is_sender_to_v<sender_2, non_sender_2>,
        "sender_2 is not a sender to non_sender_2");
    static_assert(!is_sender_to_v<sender_2, sender_2>,
        "sender_2 is not a sender to sender_2");

    static_assert(ex::is_receiver_of_v<receiver_2<int>,
                      ex::completion_signatures_of_t<sender_4<true, int>,
                          ex::env_of_t<receiver_2<int>>>>,
        "receiver_2<int> supports completion signatures of  "
        "sender_4<true,int>");
    static_assert(ex::is_receiver_of_v<example_receiver,
                      ex::completion_signatures_of_t<sender_4<true, int>,
                          ex::env_of_t<example_receiver>>>,
        "example_receiver supports completion signatures of "
        "sender_4<true,int>");
    static_assert(!ex::is_receiver_of_v<receiver_2<std::string>,
                      ex::completion_signatures_of_t<sender_4<true, int>,
                          ex::env_of_t<receiver_2<std::string>>>>,
        "receiver_2<int>  does not support completion signatures of "
        "sender_4<true,std::string>");
#ifdef HPX_HAVE_STDEXEC
    /*TODO: CHECK THAT THE RECEIVER HAVING MORE COMPLETIONS THAT THE SENDER
     * IS ACTUALLY AN ERROR IN THE OLD VERSION*/

    // It is no longer an error for the receiver to have more completions than
    // the sender
    static_assert(ex::is_receiver_of_v<receiver_2<int>,
                      ex::completion_signatures_of_t<sender_4<false, int>,
                          ex::env_of_t<receiver_2<int>>>>,
        "receiver_2<int>  does not support completion signatures of "
        "sender_4<false,int>");

    static_assert(ex::is_receiver_of_v<example_receiver,
                      ex::completion_signatures_of_t<sender_4<false, int>,
                          ex::env_of_t<example_receiver>>>,
        "example_receiver does not support completion signatures of "
        "sender_4<false,int>");

    static_assert(ex::is_receiver_of_v<example_receiver,
                      ex::completion_signatures_of_t<sender_1,
                          ex::env_of_t<example_receiver>>>,
        "example_receiver does not support completion signatures of sender_1");

#else
    static_assert(!ex::is_receiver_of_v<receiver_2<int>,
                      ex::completion_signatures_of_t<sender_4<false, int>,
                          ex::env_of_t<receiver_2<int>>>>,
        "receiver_2<int>  does not support completion signatures of "
        "sender_4<false,int>");

    static_assert(!ex::is_receiver_of_v<example_receiver,
                      ex::completion_signatures_of_t<sender_4<false, int>,
                          ex::env_of_t<example_receiver>>>,
        "example_receiver does not support completion signatures of "
        "sender_4<false,int>");

    static_assert(!ex::is_receiver_of_v<example_receiver,
                      ex::completion_signatures_of_t<sender_1,
                          ex::env_of_t<example_receiver>>>,
        "example_receiver does not support completion signatures of sender_1");
#endif

#ifdef HPX_HAVE_STDEXEC
    // Now sender_of checks for the existence of function signatures in the
    // completion signatures of the sender, not for the available set_value(...)
    // specializations.
#else
    static_assert(
        ex::is_sender_of_v<sender_1, ex::env_of_t<example_receiver>, int>,
        "sender_1 is a sender of env_of_t<example_receiver> and value types "
        "int");
    static_assert(
        ex::is_sender_of_v<sender_2, ex::env_of_t<example_receiver>, int>,
        "sender_2 is a sender of env_of_t<example_receiver> and value types "
        "int");
    static_assert(
        ex::is_sender_of_v<sender_3, ex::env_of_t<example_receiver>, int>,
        "sender_3 is a sender of env_of_t<example_receiver> and value types "
        "int");
    static_assert(ex::is_sender_of_v<sender_4<true, int>,
                      ex::env_of_t<example_receiver>, int>,
        "sender_4<true,int> is a sender of env_of_t<example_receiver> and "
        "value types "
        "int");
    static_assert(ex::is_sender_of_v<sender_4<false, int>,
                      ex::env_of_t<example_receiver>, int>,
        "sender_4<false,int> is a sender of env_of_t<example_receiver> and "
        "value types "
        "int");
    static_assert(
        ex::is_sender_of_v<sender_1, ex::env_of_t<receiver_2<int>>, int>,
        "sender_1 is a sender of env_of_t<receiver_2<int>> and value types "
        "int");
    static_assert(
        ex::is_sender_of_v<sender_2, ex::env_of_t<receiver_2<int>>, int>,
        "sender_2 is a sender of env_of_t<receiver_2<int>> and value types "
        "int");
    static_assert(
        ex::is_sender_of_v<sender_3, ex::env_of_t<receiver_2<int>>, int>,
        "sender_3 is a sender of env_of_t<receiver_2<int>> and value types "
        "int");
    static_assert(ex::is_sender_of_v<sender_4<true, std::string>,
                      ex::env_of_t<receiver_2<std::string>>, std::string>,
        "sender_4<true,std::string> is a sender of "
        "env_of_t<receiver_2<std::string>> and value types std::string");
    static_assert(ex::is_sender_of_v<sender_4<true, std::string>,
                      ex::env_of_t<receiver_2<std::string>>, std::string>,
        "sender_4<false,std::string> is a sender of "
        "env_of_t<receiver_2<std::string>> and value types std::string");
    static_assert(!ex::is_sender_of_v<sender_1,
                      ex::env_of_t<receiver_2<std::string>>, std::string>,
        "sender_1 is not a sender of env_of_t<receiver_2<std::string>> and "
        "value types std::string");
    static_assert(!ex::is_sender_of_v<sender_2,
                      ex::env_of_t<receiver_2<std::string>>, std::string>,
        "sender_2 is not a sender of env_of_t<receiver_2<std::string>> and "
        "value types std::string");
    static_assert(!ex::is_sender_of_v<sender_3,
                      ex::env_of_t<receiver_2<std::string>>, std::string>,
        "sender_3 is not a sender of env_of_t<receiver_2<std::string>> and "
        "value types std::string");
#endif

    {
        example_receiver r1;
        auto os = ex::connect(sender_1{}, r1);
        ex::start(os);
        HPX_TEST_EQ(r1.i, 4711);
        HPX_TEST_EQ(friend_tag_invoke_connect_calls, std::size_t(1));
        HPX_TEST_EQ(tag_invoke_connect_calls, std::size_t(0));
    }

    {
        example_receiver r2;
        auto os = ex::connect(sender_2{}, r2);
        ex::start(os);
        HPX_TEST_EQ(r2.i, 4711);
        HPX_TEST_EQ(friend_tag_invoke_connect_calls, std::size_t(1));
        HPX_TEST_EQ(tag_invoke_connect_calls, std::size_t(1));
    }

    {
        example_receiver r3;
        auto os = ex::connect(sender_3{}, r3);
        ex::start(os);
        HPX_TEST_EQ(r3.i, 4711);
        HPX_TEST_EQ(friend_tag_invoke_connect_calls, std::size_t(2));
        HPX_TEST_EQ(tag_invoke_connect_calls, std::size_t(1));
    }

    return hpx::util::report_errors();
}
