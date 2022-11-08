//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// we know that the sender/receiver functionalities require proper
// implementation of copy elision, and MSVC currently doesn't get
// this quite right

#include <new>
#include <optional>
#include <utility>

template <typename F>
class with_result_of_t
{
    F&& f;

public:
    using type = decltype(std::declval<F&&>()());

    explicit with_result_of_t(F&& f)
      : f(std::forward<F>(f))
    {
    }
    operator type()
    {
        return std::forward<F>(f)();
    }
};

template <typename F>
inline with_result_of_t<F> with_result_of(F&& f)
{
    return with_result_of_t<F>(std::forward<F>(f));
}

struct cant_do_anything
{
    cant_do_anything() = default;
    cant_do_anything(cant_do_anything&&) = delete;
    cant_do_anything(cant_do_anything const&) = delete;
    cant_do_anything& operator=(cant_do_anything&&) = delete;
    cant_do_anything& operator=(cant_do_anything const&) = delete;
};

cant_do_anything f()
{
    return {};
}

int main()
{
    std::optional<cant_do_anything> value;
    value.emplace(with_result_of(&f));
}
