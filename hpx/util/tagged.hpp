//  Copyright Eric Niebler 2013-2015
//  Copyright 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This was modeled after the code available in the Range v3 library

#if !defined(HPX_UTIL_TAGGED_DEC_23_2015_1014AM)
#define HPX_UTIL_TAGGED_DEC_23_2015_1014AM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/tuple.hpp>

#include <functional>

namespace hpx { namespace util
{
    template <typename Base, typename ...Tags>
    struct tagged;

    namespace tag
    {
        struct specifier_tag {};
    }

    /// \cond
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        struct getters
        {
        private:
            template <typename, typename...>
            friend struct util::tagged;

            template <typename T, typename Tag, std::size_t I>
            struct unpack_getter
            {
                typedef typename Tag::template getter<
                        T, typename tuple_element<I, T>::type, I
                    > type;

            private:
                template <typename, typename...>
                friend struct util::tagged;
            };

            template <typename T, typename Indices, typename ...Tags>
            struct collect_;

            template <typename T, std::size_t ...Is, typename ...Tags>
            struct collect_<T, detail::pack_c<std::size_t, Is...>, Tags...>
              : unpack_getter<T, Tags, Is>::type...
            {};

        public:
            template <typename T, typename ...Tags>
            struct collect
              : collect_<
                    T,
                    typename detail::make_index_pack<sizeof...(Tags)>::type,
                    Tags...
                >
            {};
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct tag_spec;

        template <typename Spec, typename Elem>
        struct tag_spec<Spec(Elem)>
        {
            typedef Spec type;
        };

        template <typename T>
        struct tag_elem;

        template <typename Spec, typename Elem>
        struct tag_elem<Spec(Elem)>
        {
            typedef Elem type;
        };
    }
    /// \endcond

    template <typename Base, typename ...Tags>
    struct tagged
      : Base, detail::getters::collect<tagged<Base, Tags...>, Tags...>
    {
        // specified as:
        //
        // using Base::Base;
        // tagged() = default;
        // tagged(tagged&&) = default;
        // tagged(const tagged&) = default;
        // tagged &operator=(tagged&&) = default;
        // tagged &operator=(const tagged&) = default;

        template <typename ... Ts>
        tagged(Ts && ... ts)
          : Base(std::forward<Ts>(ts)...)
        {}

        // Effects: Initializes Base with static_cast<Other &&>(rhs).
        // Returns: *this.
        template <typename Other,
        HPX_CONCEPT_REQUIRES_(
            std::is_convertible<Other, Base>::value)>
        tagged(tagged<Other, Tags...> && rhs) HPX_NOEXCEPT
          : Base(static_cast<Other&&>(rhs))
        {}

        // Effects: Initializes Base with static_cast<Other const&>(rhs).
        // Returns: *this.
        template <typename Other,
        HPX_CONCEPT_REQUIRES_(
            std::is_convertible<Other, Base>::value)>
        tagged(tagged<Other, Tags...> const& rhs)
          : Base(static_cast<Other const&>(rhs))
        {}

        // Effects: Assigns static_cast<Other&&>(rhs) to
        //          static_cast<Base&>(*this).
        // Returns: *this.
        template<typename Other,
        HPX_CONCEPT_REQUIRES_(
            std::is_convertible<Other, Base>::value)>
        tagged &operator=(tagged<Other, Tags...> && rhs)
        {
            static_cast<Base&>(*this) = static_cast<Other&&>(rhs);
            return *this;
        }

        // Effects: Assigns static_cast<Other const&>(rhs) to
        //          static_cast<Base&>(*this).
        // Returns: *this.
        template<typename Other,
        HPX_CONCEPT_REQUIRES_(
            std::is_convertible<Other, Base>::value)>
        tagged &operator=(tagged<Other, Tags...> const& rhs)
        {
            static_cast<Base&>(*this) = static_cast<Other const&>(rhs);
            return *this;
        }

        // Effects: Assigns std::forward<U>(u) to static_cast<Base&>(*this).
        // Returns: *this.
        template<typename U,
        HPX_CONCEPT_REQUIRES_(
            !std::is_same<tagged, typename decay<U>::type>::value &&
            std::is_convertible<U, Base>::value)>
        tagged &operator=(U && u)
        {
            static_cast<Base&>(*this) = std::forward<U>(u);
            return *this;
        }

        // Effects: Calls swap on the result of applying static_cast to *this
        //         and other.
        // Throws: Nothing unless the call to swap on the Base sub-objects
        //         throws.
        HPX_FORCEINLINE void swap(tagged &other) HPX_NOEXCEPT
        {
            std::swap(static_cast<Base &>(*this), static_cast<Base &>(other));
        }

        // Effects: x.swap(y).
        // Throws: Nothing unless the call to x.swap(y) throws.
        friend HPX_FORCEINLINE void swap(tagged &x, tagged &y) HPX_NOEXCEPT
        {
            x.swap(y);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Base, typename...Tags>
    struct tuple_size<tagged<Base, Tags...> >
      : tuple_size<Base>
    {};

    template <size_t N, typename Base, typename...Tags>
    struct tuple_element<N, tagged<Base, Tags...> >
      : tuple_element<N, Base>
    {};
}}

// A tagged getter is an empty trivial class type that has a named member
// function that returns a reference to a member of a tuple-like object that is
// assumed to be derived from the getter class. The tuple-like type of a tagged
// getter is called its DerivedCharacteristic. The index of the tuple element
// returned from the getter's member functions is called its ElementIndex. The
// name of the getter's member function is called its ElementName.

// A tagged getter class with DerivedCharacteristic D, ElementIndex N, and
// ElementName name shall provide the following interface:
//
//     struct __TAGGED_GETTER
//     {
//         constexpr decltype(auto)
//             name() & { return get<N>(static_cast<D&>(*this)); }
//         constexpr decltype(auto)
//             name() && { return get<N>(static_cast<D&&>(*this)); }
//         constexpr decltype(auto)
//             name() const & { return get<N>(static_cast<const D&>(*this)); }
//     };

#define HPX_DEFINE_TAG_SPECIFIER(NAME)                                        \
namespace tag                                                                 \
{                                                                             \
    struct NAME : hpx::util::tag::specifier_tag                               \
    {                                                                         \
    private:                                                                  \
        friend struct hpx::util::detail::getters;                             \
                                                                              \
        template <typename Derived, typename Type, std::size_t I>             \
        struct getter                                                         \
        {                                                                     \
            HPX_FORCEINLINE Type& NAME()                                      \
            {                                                                 \
                return hpx::util::get<I>(static_cast<Derived&>(*this));       \
            }                                                                 \
            HPX_CONSTEXPR HPX_FORCEINLINE Type const& NAME() const            \
            {                                                                 \
                return hpx::util::get<I>(static_cast<Derived const&>(*this)); \
            }                                                                 \
                                                                              \
        private:                                                              \
            friend struct hpx::util::detail::getters;                         \
        };                                                                    \
    };                                                                        \
}                                                                             \
/**/

///////////////////////////////////////////////////////////////////////////////
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wunknown-pragmas"
#  pragma clang diagnostic ignored "-Wpragmas"
#  pragma clang diagnostic ignored "-Wmismatched-tags"
#endif

#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#  if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#    pragma GCC diagnostic push
#  endif
#  pragma GCC diagnostic ignored "-Wunknown-pragmas"
#  pragma GCC diagnostic ignored "-Wpragmas"
#  pragma GCC diagnostic ignored "-Wmismatched-tags"
#endif

namespace std
{
    template <typename Base, typename...Tags>
    struct tuple_size<hpx::util::tagged<Base, Tags...> >
      : tuple_size<Base>
    {};

    template <size_t N, typename Base, typename...Tags>
    struct tuple_element<N, hpx::util::tagged<Base, Tags...> >
      : tuple_element<N, Base>
    {};
}

#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#  if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#    pragma GCC diagnostic pop
#  endif
#endif

#if defined(__clang__)
#  pragma clang diagnostic pop
#endif

#endif
