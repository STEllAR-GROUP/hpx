//  Copyright (c) 2017-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/construct_at.hpp>

#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx::optional_ns {

    struct nullopt_t
    {
        struct init
        {
        };
        constexpr explicit nullopt_t(nullopt_t::init) noexcept {}
    };
    constexpr nullopt_t nullopt{nullopt_t::init()};

    class HPX_ALWAYS_EXPORT bad_optional_access : public std::logic_error
    {
    public:
        explicit bad_optional_access(std::string const& what_arg)
          : std::logic_error(what_arg)
        {
        }

        explicit bad_optional_access(char const* what_arg)
          : std::logic_error(what_arg)
        {
        }
    };

    namespace _optional_swap {

        using std::swap;

        template <typename T>
        void check_swap() noexcept(
            noexcept(swap(std::declval<T&>(), std::declval<T&>())));
    }    // namespace _optional_swap

    template <typename T>
    class optional
    {
    public:
        using value_type = T;

        constexpr optional() noexcept
          : empty_(true)
        {
        }

        constexpr optional(nullopt_t) noexcept
          : empty_(true)
        {
        }

        optional(optional const& other)
          : empty_(true)
        {
            if (!other.empty_)
            {
                hpx::construct_at(
                    reinterpret_cast<T*>(&storage_), other.value());
                empty_ = false;
            }
        }
        optional(optional&& other) noexcept(
            std::is_nothrow_move_constructible_v<T>)
          : empty_(true)
        {
            if (!other.empty_)
            {
                hpx::construct_at(
                    reinterpret_cast<T*>(&storage_), HPX_MOVE(other.value()));
                empty_ = false;
            }
        }

        template <typename U = T,
            typename = std::enable_if_t<std::is_constructible_v<T, U&&>>>
        explicit optional(U&& val) noexcept(
            std::is_nothrow_move_constructible_v<T>)
          : empty_(true)
        {
            hpx::construct_at(reinterpret_cast<T*>(&storage_), HPX_MOVE(val));
            empty_ = false;
        }

        template <typename U,
            typename = std::enable_if_t<std::is_constructible_v<T, U const&>>>
        constexpr optional(optional<U> const& val)
          : empty_(true)
        {
            if (val.has_value())
            {
                hpx::construct_at(reinterpret_cast<T*>(&storage_), *val);
                empty_ = false;
            }
        }
        template <typename U,
            typename = std::enable_if_t<std::is_constructible_v<T, U&&>>>
        constexpr optional(optional<U>&& val) noexcept(
            std::is_nothrow_move_constructible_v<T>)
          : empty_(true)
        {
            if (val.has_value())
            {
                hpx::construct_at(
                    reinterpret_cast<T*>(&storage_), HPX_MOVE(*val));
                empty_ = false;
            }
        }

        template <typename... Ts>
        explicit optional(std::in_place_t, Ts&&... ts)
          : empty_(true)
        {
            hpx::construct_at(
                reinterpret_cast<T*>(&storage_), HPX_FORWARD(Ts, ts)...);
            empty_ = false;
        }

        template <typename U, typename... Ts>
        explicit optional(
            std::in_place_t, std::initializer_list<U> il, Ts&&... ts)
          : empty_(true)
        {
            hpx::construct_at(
                reinterpret_cast<T*>(&storage_), il, HPX_FORWARD(Ts, ts)...);
            empty_ = false;
        }

        ~optional()
        {
            reset();
        }

        optional& operator=(optional const& other)
        {
            optional tmp(other);
            swap(tmp);

            return *this;
        }
        optional& operator=(optional&& other) noexcept(
            std::is_nothrow_move_assignable_v<T>&&
                std::is_nothrow_move_constructible_v<T>)
        {
            if (this == &other)
            {
                return *this;
            }

            if (empty_)
            {
                if (!other.empty_)
                {
                    hpx::construct_at(reinterpret_cast<T*>(&storage_),
                        HPX_MOVE(other.value()));
                    empty_ = false;
                }
            }
            else
            {
                if (other.empty_)
                {
                    std::destroy_at(reinterpret_cast<T*>(&storage_));
                    empty_ = true;
                }
                else
                {
                    **this = HPX_MOVE(other.value());
                }
            }
            return *this;
        }

        optional& operator=(T&& other)
        {
            if (!empty_)
            {
                std::destroy_at(reinterpret_cast<T*>(&storage_));
                empty_ = true;
            }

            hpx::construct_at(reinterpret_cast<T*>(&storage_), HPX_MOVE(other));
            empty_ = false;

            return *this;
        }
        template <typename U,
            typename Enable =
                std::enable_if_t<std::is_constructible_v<T, U const&>>>
        optional& operator=(optional<U> const& other)
        {
            optional tmp(other);
            swap(tmp);

            return *this;
        }
        template <typename U,
            typename Enable = std::enable_if_t<std::is_constructible_v<T, U&&>>>
        optional& operator=(optional<U>&& other)
        {
            optional tmp(HPX_MOVE(other));
            swap(tmp);

            return *this;
        }

        optional& operator=(nullopt_t) noexcept
        {
            if (!empty_)
            {
                std::destroy_at(reinterpret_cast<T*>(&storage_));
                empty_ = true;
            }
            return *this;
        }

        ///////////////////////////////////////////////////////////////////////
        constexpr T const* operator->() const noexcept
        {
            return reinterpret_cast<T const*>(&storage_);
        }

        T* operator->() noexcept
        {
            return reinterpret_cast<T*>(&storage_);
        }

        constexpr T const& operator*() const& noexcept
        {
            return *reinterpret_cast<T const*>(&storage_);
        }
        T& operator*() & noexcept
        {
            return *reinterpret_cast<T*>(&storage_);
        }
        T&& operator*() && noexcept
        {
            return HPX_MOVE(*reinterpret_cast<T*>(&storage_));
        }
        T const&& operator*() const&& noexcept
        {
            return HPX_MOVE(*reinterpret_cast<T*>(&storage_));
        }

        [[nodiscard]] constexpr explicit operator bool() const noexcept
        {
            return !empty_;
        }

        [[nodiscard]] constexpr bool has_value() const
        {
            return !empty_;
        }

        T& value() &
        {
            if (empty_)
            {
                throw bad_optional_access(
                    "object is empty during call to 'value()'");
            }
            return **this;
        }
        T&& value() &&
        {
            if (empty_)
            {
                throw bad_optional_access(
                    "object is empty during call to 'value()'");
            }
            return **this;
        }

        constexpr T const& value() const&
        {
            if (empty_)
            {
                throw bad_optional_access(
                    "object is empty during call to 'value()'");
            }
            return HPX_MOVE(**this);
        }
        constexpr T const&& value() const&&
        {
            if (empty_)
            {
                throw bad_optional_access(
                    "object is empty during call to 'value()'");
            }
            return HPX_MOVE(**this);
        }

        template <typename U>
        constexpr T value_or(U&& value) const&
        {
            if (empty_)
                return HPX_FORWARD(U, value);
            return **this;
        }
        template <typename U>
        T value_or(U&& value) &&
        {
            if (empty_)
                return HPX_FORWARD(U, value);
            return HPX_MOVE(**this);
        }

        template <typename... Ts>
        void emplace(Ts&&... ts)
        {
            if (!empty_)
            {
                std::destroy_at(reinterpret_cast<T*>(&storage_));
                empty_ = true;
            }
            hpx::construct_at(
                reinterpret_cast<T*>(&storage_), HPX_FORWARD(Ts, ts)...);
            empty_ = false;
        }

#if !defined(HPX_HAVE_CXX17_COPY_ELISION) ||                                   \
    !defined(HPX_HAVE_CXX17_OPTIONAL_COPY_ELISION)
        // workaround for broken return type copy elision in MSVC
        template <typename F, typename... Ts>
        void emplace_f(F&& f, Ts&&... ts)
        {
            if (!empty_)
            {
                std::destroy_at(reinterpret_cast<T*>(&storage_));
                empty_ = true;
            }
            std::construct_at(reinterpret_cast<T*>(&storage_),
                HPX_FORWARD(F, f)(HPX_FORWARD(Ts, ts)...));
            empty_ = false;
        }
#endif

        void swap(optional& other) noexcept(
            std::is_nothrow_move_constructible_v<T>&& noexcept(
                _optional_swap::check_swap<T>()))
        {
            // do nothing if both are empty
            if (empty_ && other.empty_)
            {
                return;
            }

            // swap content if both are non-empty
            if (!empty_ && !other.empty_)
            {
                using std::swap;
                swap(**this, *other);
                return;
            }

            // move the non-empty one into the empty one and make remains empty
            optional* empty = empty_ ? this : &other;
            optional* non_empty = empty_ ? &other : this;

            hpx::construct_at(
                reinterpret_cast<T*>(&empty->storage_), HPX_MOVE(**non_empty));
            std::destroy_at(reinterpret_cast<T*>(&non_empty->storage_));

            empty->empty_ = false;
            non_empty->empty_ = true;
        }

        void reset() noexcept
        {
            if (!empty_)
            {
                std::destroy_at(reinterpret_cast<T*>(&storage_));
                empty_ = true;
            }
        }

        template <typename F>
        auto and_then(F&& f) &
        {
            if (*this)
            {
                // using std::invoke to avoid circular dependencies
                return std::invoke(HPX_FORWARD(F, f), value());
            }

            using result_type = std::invoke_result_t<F, decltype(value())>;
            return std::remove_cv_t<std::remove_reference_t<result_type>>();
        }

        template <typename F>
        constexpr auto and_then(F&& f) const&
        {
            if (*this)
            {
                // using std::invoke to avoid circular dependencies
                return std::invoke(HPX_FORWARD(F, f), value());
            }

            using result_type = std::invoke_result_t<F, decltype(value())>;
            return std::remove_cv_t<std::remove_reference_t<result_type>>();
        }

        template <typename F>
        auto and_then(F&& f) &&
        {
            if (*this)
            {
                // using std::invoke to avoid circular dependencies
                return std::invoke(HPX_FORWARD(F, f), HPX_MOVE(value()));
            }

            using result_type =
                std::invoke_result_t<F, decltype(HPX_MOVE(value()))>;
            return std::remove_cv_t<std::remove_reference_t<result_type>>();
        }

        template <typename F>
        auto and_then(F&& f) const&&
        {
            if (*this)
            {
                // using std::invoke to avoid circular dependencies
                return std::invoke(HPX_FORWARD(F, f), HPX_MOVE(value()));
            }

            using result_type =
                std::invoke_result_t<F, decltype(HPX_MOVE(value()))>;
            return std::remove_cv_t<std::remove_reference_t<result_type>>();
        }

        template <typename F>
        auto transform(F&& f) &
        {
            using result_type = std::invoke_result_t<F, decltype(value())>;
            if (*this)
            {
                // using std::invoke to avoid circular dependencies
                return optional<result_type>(
                    std::invoke(HPX_FORWARD(F, f), value()));
            }
            return optional<result_type>();
        }

        template <typename F>
        constexpr auto transform(F&& f) const&
        {
            using result_type = std::invoke_result_t<F, decltype(value())>;
            if (*this)
            {
                // using std::invoke to avoid circular dependencies
                return optional<result_type>(
                    std::invoke(HPX_FORWARD(F, f), value()));
            }
            return optional<result_type>();
        }

        template <typename F>
        auto transform(F&& f) &&
        {
            using result_type =
                std::invoke_result_t<F, decltype(HPX_MOVE(value()))>;
            if (*this)
            {
                // using std::invoke to avoid circular dependencies
                return optional<result_type>(
                    std::invoke(HPX_FORWARD(F, f), HPX_MOVE(value())));
            }
            return optional<result_type>();
        }

        template <typename F>
        auto transform(F&& f) const&&
        {
            using result_type =
                std::invoke_result_t<F, decltype(HPX_MOVE(value()))>;
            if (*this)
            {
                // using std::invoke to avoid circular dependencies
                return optional<result_type>(
                    std::invoke(HPX_FORWARD(F, f), HPX_MOVE(value())));
            }
            return optional<result_type>();
        }

        template <typename F,
            typename = std::enable_if_t<std::is_invocable_v<F> &&
                std::is_copy_constructible_v<T>>>
        constexpr optional or_else(F&& f) const&
        {
            if (*this)
            {
                return *this;
            }
            return {HPX_FORWARD(F, f)()};
        }

        template <typename F,
            typename = std::enable_if_t<std::is_invocable_v<F> &&
                std::is_move_constructible_v<T>>>
        optional or_else(F&& f) &&
        {
            if (*this)
            {
                return HPX_MOVE(*this);
            }
            return {HPX_FORWARD(F, f)()};
        }

    private:
        std::aligned_storage_t<sizeof(T), alignof(T)> storage_;
        bool empty_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    [[nodiscard]] constexpr bool operator==(
        optional<T> const& lhs, optional<T> const& rhs)
    {
        // different clang-format versions disagree
        // clang-format off
        return (static_cast<bool>(lhs) != static_cast<bool>(rhs)) ? false :
            (!static_cast<bool>(lhs) && !static_cast<bool>(rhs))  ? true :
                                          (*lhs == *rhs);
        // clang-format on
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator!=(
        optional<T> const& lhs, optional<T> const& rhs)
    {
        return !(lhs == rhs);
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator<(
        optional<T> const& lhs, optional<T> const& rhs)
    {
        return (!static_cast<bool>(rhs)) ? false :
            (!static_cast<bool>(lhs))    ? true :
                                           *rhs < *lhs;
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator>=(
        optional<T> const& lhs, optional<T> const& rhs)
    {
        return !(lhs < rhs);
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator>(
        optional<T> const& lhs, optional<T> const& rhs)
    {
        return (!static_cast<bool>(lhs)) ? false :
            (!static_cast<bool>(rhs))    ? true :
                                           *rhs > *lhs;
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator<=(
        optional<T> const& lhs, optional<T> const& rhs)
    {
        return !(lhs > rhs);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    [[nodiscard]] constexpr bool operator==(
        optional<T> const& opt, nullopt_t) noexcept
    {
        return !static_cast<bool>(opt);
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator==(
        nullopt_t, optional<T> const& opt) noexcept
    {
        return !static_cast<bool>(opt);
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator!=(
        optional<T> const& opt, nullopt_t) noexcept
    {
        return static_cast<bool>(opt);
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator!=(
        nullopt_t, optional<T> const& opt) noexcept
    {
        return static_cast<bool>(opt);
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator<(
        optional<T> const&, nullopt_t) noexcept
    {
        return false;
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator<(
        nullopt_t, optional<T> const& opt) noexcept
    {
        return static_cast<bool>(opt);
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator>=(
        optional<T> const&, nullopt_t) noexcept
    {
        return true;
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator>=(
        nullopt_t, optional<T> const& opt) noexcept
    {
        return !static_cast<bool>(opt);
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator>(
        optional<T> const& opt, nullopt_t) noexcept
    {
        return static_cast<bool>(opt);
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator>(
        nullopt_t, optional<T> const&) noexcept
    {
        return false;
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator<=(
        optional<T> const& opt, nullopt_t) noexcept
    {
        return !static_cast<bool>(opt);
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator<=(
        nullopt_t, optional<T> const&) noexcept
    {
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    [[nodiscard]] constexpr bool operator==(
        optional<T> const& opt, T const& value)
    {
        return static_cast<bool>(opt) ? (*opt == value) : false;
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator==(
        T const& value, optional<T> const& opt)
    {
        return static_cast<bool>(opt) ? (value == *opt) : false;
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator!=(
        optional<T> const& opt, T const& value)
    {
        return !(opt == value);
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator!=(
        T const& value, optional<T> const& opt)
    {
        return !(value == *opt);
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator<(
        optional<T> const& opt, T const& value)
    {
        return static_cast<bool>(opt) ? (*opt < value) : true;
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator<(
        T const& value, optional<T> const& opt)
    {
        return static_cast<bool>(opt) ? (value < *opt) : false;
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator>=(
        optional<T> const& opt, T const& value)
    {
        return !(*opt < value);
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator>=(
        T const& value, optional<T> const& opt)
    {
        return !(value < *opt);
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator>(
        optional<T> const& opt, T const& value)
    {
        return static_cast<bool>(opt) ? (*opt > value) : false;
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator>(
        T const& value, optional<T> const& opt)
    {
        return static_cast<bool>(opt) ? (value > *opt) : true;
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator<=(
        optional<T> const& opt, T const& value)
    {
        return !(*opt > value);
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator<=(
        T const& value, optional<T> const& opt)
    {
        return !(value > *opt);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    void swap(optional<T>& x, optional<T>& y) noexcept(noexcept(x.swap(y)))
    {
        x.swap(y);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    constexpr optional<std::decay_t<T>> make_optional(T&& v)
    {
        return optional<std::decay_t<T>>(HPX_FORWARD(T, v));
    }

    template <typename T, typename... Ts>
    constexpr optional<T> make_optional(Ts&&... ts)
    {
        return optional<T>(std::in_place, HPX_FORWARD(Ts, ts)...);
    }

    template <typename T, typename U, typename... Ts>
    constexpr optional<T> make_optional(std::initializer_list<U> il, Ts&&... ts)
    {
        return optional<T>(std::in_place, il, HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx::optional_ns

///////////////////////////////////////////////////////////////////////////////
namespace std {

    template <typename T>
    struct hash<::hpx::optional_ns::optional<T>>
    {
        constexpr std::size_t operator()(
            ::hpx::optional_ns::optional<T> const& arg) const
        {
            return arg ? std::hash<T>{}(*arg) : std::size_t{};
        }
    };
}    // namespace std
