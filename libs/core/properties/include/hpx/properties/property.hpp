#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/tag_override_invoke.hpp>
#include <hpx/type_support/always_void.hpp>

#include <iostream>
#include <type_traits>

namespace hpx { namespace experimental {
    template <typename T, typename Property>
    struct is_applicable_property
    {
        static constexpr bool value =
            Property::template is_applicable_property_v<T>;
    };

    template <typename T, typename Property>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_applicable_property_v =
        is_applicable_property<typename std::decay<T>::type,
            typename std::decay<Property>::type>::value;

    namespace detail {
        template <typename T, typename Property, typename Enable = void>
        struct has_static_value : std::false_type
        {
        };

        template <typename T, typename Property>
        struct has_static_value<T, Property,
            typename hpx::util::always_void<decltype(
                std::decay<Property>::type::template static_query_v<
                    typename std::decay<T>::type> ==
                std::decay<Property>::type::value())>::type> : std::true_type
        {
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct require_concept_t
      : hpx::functional::tag_override<require_concept_t>
    {
        template <typename T, typename Property>
        friend constexpr HPX_FORCEINLINE auto
        tag_override_invoke(require_concept_t, T&& t, Property&& p) noexcept(
            noexcept(
                std::forward<T>(t).require_concept(std::forward<Property>(p))))
            -> typename std::enable_if<is_applicable_property_v<T, Property> &&
                    std::decay<Property>::type::is_requirable_concept &&
                    detail::has_static_value<T, Property>::value,
                decltype(std::forward<T>(t))>::type
        {
            return std::forward<T>(t);
        }

        template <typename T, typename Property>
        friend constexpr HPX_FORCEINLINE auto
        tag_override_invoke(require_concept_t, T&& t, Property&& p) noexcept(
            noexcept(
                std::forward<T>(t).require_concept(std::forward<Property>(p))))
            -> typename std::enable_if<is_applicable_property_v<T, Property> &&
                    std::decay<Property>::type::is_requirable_concept &&
                    !detail::has_static_value<T, Property>::value,
                decltype(std::forward<T>(t).require_concept(
                    std::forward<Property>(p)))>::type
        {
            return std::forward<T>(t).require_concept(
                std::forward<Property>(p));
        }
    } require_concept;

    HPX_INLINE_CONSTEXPR_VARIABLE struct require_t
      : hpx::functional::tag_override<require_t>
    {
        template <typename T, typename Property>
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(require_t,
            T&& t, Property&& p) noexcept(noexcept(std::forward<T>(t))) ->
            typename std::enable_if<is_applicable_property_v<T, Property> &&
                    std::decay<Property>::type::is_requirable &&
                    detail::has_static_value<T, Property>::value,
                decltype(std::forward<T>(t))>::type
        {
            return std::forward<T>(t);
        }

        template <typename T, typename Property>
        friend constexpr HPX_FORCEINLINE auto
        tag_override_invoke(require_t, T&& t, Property&& p) noexcept(
            noexcept(std::forward<T>(t).require(std::forward<Property>(p)))) ->
            typename std::enable_if<is_applicable_property_v<T, Property> &&
                    std::decay<Property>::type::is_requirable &&
                    !detail::has_static_value<T, Property>::value,
                decltype(std::forward<T>(t).require(
                    std::forward<Property>(p)))>::type
        {
            return std::forward<T>(t).require(std::forward<Property>(p));
        }

        template <typename T, typename Property0, typename Property1,
            typename... Properties>
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(require_t,
            T&& t, Property0&& p0, Property1&& p1,
            Properties&&... ps) noexcept(noexcept(require_t{}(require_t{}(std::
                                                                              forward<
                                                                                  T>(
                                                                                  t),
                                                                  std::forward<
                                                                      Property0>(
                                                                      p0)),
            std::forward<Property1>(p1), std::forward<Properties>(ps)...))) ->
            typename std::enable_if<is_applicable_property_v<T, Property0> &&
                    std::decay<Property0>::type::is_requirable,
                decltype(require_t{}(require_t{}(std::forward<T>(t),
                                         std::forward<Property0>(p0)),
                    std::forward<Property1>(p1),
                    std::forward<Properties>(ps)...))>::type
        {
            return require_t{}(
                require_t{}(std::forward<T>(t), std::forward<Property0>(p0)),
                std::forward<Property1>(p1), std::forward<Properties>(ps)...);
        }
    } require;

    namespace detail {
        template <typename T, typename Property, typename Enable = void>
        struct has_member_require : std::false_type
        {
        };

        template <typename T, typename Property>
        struct has_member_require<T, Property,
            typename hpx::util::always_void<decltype(std::declval<T>().require(
                std::declval<Property>()))>::type> : std::true_type
        {
        };

        template <typename T, typename Property, typename Enable = void>
        struct has_tag_invoke_require : std::false_type
        {
        };

        template <typename T, typename Property>
        struct has_tag_invoke_require<T, Property,
            typename hpx::util::always_void<decltype(tag_invoke(require_t{},
                std::declval<T>(), std::declval<Property>()))>::type>
          : std::true_type
        {
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct prefer_t
      : hpx::functional::tag_override<prefer_t>
    {
        template <typename T, typename Property>
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(prefer_t,
            T&& t, Property&& p) noexcept(noexcept(std::forward<T>(t))) ->
            typename std::enable_if<is_applicable_property_v<T, Property> &&
                    std::decay<Property>::type::is_preferable &&
                    detail::has_static_value<T, Property>::value,
                decltype(std::forward<T>(t))>::type
        {
            return std::forward<T>(t);
        }

        template <typename T, typename Property>
        friend constexpr HPX_FORCEINLINE auto
        tag_override_invoke(prefer_t, T&& t, Property&& p) noexcept(
            noexcept(std::forward<T>(t).require(std::forward<Property>(p)))) ->
            typename std::enable_if<is_applicable_property_v<T, Property> &&
                    std::decay<Property>::type::is_preferable &&
                    !detail::has_static_value<T, Property>::value,
                decltype(std::forward<T>(t).require(
                    std::forward<Property>(p)))>::type
        {
            return std::forward<T>(t).require(std::forward<Property>(p));
        }

        template <typename T, typename Property>
        friend constexpr HPX_FORCEINLINE auto tag_invoke(prefer_t, T&& t,
            Property&& p) noexcept(noexcept(std::forward<T>(t))) ->
            typename std::enable_if<is_applicable_property_v<T, Property> &&
                    std::decay<Property>::type::is_preferable &&
                    !detail::has_static_value<T, Property>::value &&
                    !detail::has_member_require<T, Property>::value &&
                    !detail::has_tag_invoke_require<T, Property>::value,
                decltype(std::forward<T>(t))>::type
        {
            return std::forward<T>(t);
        }

        template <typename T, typename Property0, typename Property1,
            typename... Properties>
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(prefer_t,
            T&& t, Property0&& p0, Property1&& p1,
            Properties&&... ps) noexcept(noexcept(prefer_t{}(prefer_t{}(std::
                                                                            forward<
                                                                                T>(
                                                                                t),
                                                                 std::forward<
                                                                     Property0>(
                                                                     p0)),
            std::forward<Property1>(p1), std::forward<Properties>(ps)...))) ->
            typename std::enable_if<is_applicable_property_v<T, Property0> &&
                    std::decay<Property0>::type::is_preferable,
                decltype(prefer_t{}(
                    prefer_t{}(std::forward<T>(t), std::forward<Property0>(p0)),
                    std::forward<Property1>(p1),
                    std::forward<Properties>(ps)...))>::type
        {
            return prefer_t{}(
                prefer_t{}(std::forward<T>(t), std::forward<Property0>(p0)),
                std::forward<Property1>(p1), std::forward<Properties>(ps)...);
        }
    } prefer;

    namespace detail {
        template <typename T, typename Property, typename Enable = void>
        struct has_static_query : std::false_type
        {
        };

        template <typename T, typename Property>
        struct has_static_query<T, Property,
            typename hpx::util::always_void<decltype(
                std::decay<Property>::type::template static_query_v<
                    typename std::decay<T>::type>)>::type> : std::true_type
        {
        };

    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct query_t
      : hpx::functional::tag_override<query_t>
    {
        template <typename T, typename Property>
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(query_t,
            T&& t, Property&& p) noexcept(noexcept(std::decay<Property>::type::
                template static_query_v<typename std::decay<T>::type>)) ->
            typename std::enable_if<is_applicable_property_v<T, Property> &&
                    detail::has_static_query<T, Property>::value,
                decltype(std::decay<Property>::type::template static_query_v<
                    typename std::decay<T>::type>)>::type
        {
            return std::decay<Property>::type::template static_query_v<
                typename std::decay<T>::type>;
        }

        template <typename T, typename Property>
        friend constexpr HPX_FORCEINLINE auto
        tag_override_invoke(query_t, T&& t, Property&& p) noexcept(
            noexcept(std::forward<T>(t).query(std::forward<Property>(p)))) ->
            typename std::enable_if<
                is_applicable_property_v<typename std::decay<T>::type,
                    typename std::decay<Property>::type> &&
                    !detail::has_static_query<T, Property>::value,
                decltype(
                    std::forward<T>(t).query(std::forward<Property>(p)))>::type
        {
            return std::forward<T>(t).query(std::forward<Property>(p));
        }
    } query;

    template <typename T, typename Property>
    struct can_require_concept
      : std::integral_constant<bool,
            hpx::traits::is_invocable<require_concept_t, T, Property>::value>
    {
    };

    template <typename T, typename... Properties>
    struct can_require
      : std::integral_constant<bool,
            hpx::traits::is_invocable<require_t, T, Properties...>::value>
    {
    };

    template <typename T, typename... Properties>
    struct can_prefer
      : std::integral_constant<bool,
            hpx::traits::is_invocable<prefer_t, T, Properties...>::value>
    {
    };

    template <typename T, typename Property>
    struct can_query
      : std::integral_constant<bool,
            hpx::traits::is_invocable<query_t, T, Property>::value>
    {
    };

    template <typename T, typename Property>
    HPX_INLINE_CONSTEXPR_VARIABLE bool can_require_concept_v =
        can_require_concept<T, Property>::value;

    template <typename T, typename... Properties>
    HPX_INLINE_CONSTEXPR_VARIABLE bool can_require_v =
        can_require<T, Properties...>::value;

    template <typename T, typename... Properties>
    HPX_INLINE_CONSTEXPR_VARIABLE bool can_prefer_v =
        can_prefer<T, Properties...>::value;

    template <typename T, typename Property>
    HPX_INLINE_CONSTEXPR_VARIABLE bool can_query_v =
        can_query<T, Property>::value;
}}    // namespace hpx::experimental
