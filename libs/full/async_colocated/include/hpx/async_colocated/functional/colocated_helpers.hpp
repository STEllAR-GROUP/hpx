//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/agas_base/gva.hpp>
#include <hpx/async_distributed/continuation.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/unique_ptr.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/unused.hpp>

#include <memory>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace functional {

    ///////////////////////////////////////////////////////////////////////////
    struct extract_locality
    {
        hpx::id_type operator()(
            hpx::id_type const& locality_id, hpx::id_type const& id) const
        {
            if (locality_id == hpx::invalid_id)
            {
                HPX_THROW_EXCEPTION(hpx::no_success,
                    "extract_locality::operator()",
                    "could not resolve colocated locality for id({1})", id);
                return hpx::invalid_id;
            }
            return locality_id;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename Bound, typename Continuation>
        struct post_continuation_impl
        {
            typedef typename std::decay<Bound>::type bound_type;
            typedef typename std::decay<Continuation>::type continuation_type;

            post_continuation_impl() = default;

            template <typename Bound_, typename Continuation_>
            explicit post_continuation_impl(Bound_&& bound, Continuation_&& c)
              : bound_(HPX_FORWARD(Bound_, bound))
              , cont_(HPX_FORWARD(Continuation_, c))
            {
            }

            post_continuation_impl(post_continuation_impl&& o) = default;

            post_continuation_impl& operator=(
                post_continuation_impl&& o) = default;

            template <typename T>
            typename util::invoke_result<bound_type, hpx::id_type, T>::type
            operator()(hpx::id_type lco, T&& t)
            {
                typedef typename util::invoke_result<bound_type, hpx::id_type,
                    T>::type result_type;

                bound_.post_c(HPX_MOVE(cont_), lco, HPX_FORWARD(T, t));
                return result_type();
            }

        private:
            // serialization support
            friend class hpx::serialization::access;

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const)
            {
                // clang-format off
                ar & bound_ & cont_;
                // clang-format on
            }

            bound_type bound_;
            continuation_type cont_;
        };

        template <typename Bound>
        struct post_continuation_impl<Bound, hpx::util::unused_type>
        {
            typedef typename std::decay<Bound>::type bound_type;

            post_continuation_impl() = default;

            template <typename Bound_,
                typename Enable = typename std::enable_if<
                    !std::is_same<typename std::decay<Bound_>::type,
                        post_continuation_impl>::value>::type>
            explicit post_continuation_impl(Bound_&& bound)
              : bound_(HPX_FORWARD(Bound_, bound))
            {
            }

            post_continuation_impl(post_continuation_impl&& o) = default;

            post_continuation_impl& operator=(
                post_continuation_impl&& o) = default;

            template <typename T>
            typename util::invoke_result<bound_type, hpx::id_type, T>::type
            operator()(hpx::id_type lco, T&& t)
            {
                typedef typename util::invoke_result<bound_type, hpx::id_type,
                    T>::type result_type;

                bound_.post(lco, HPX_FORWARD(T, t));
                return result_type();
            }

        private:
            // serialization support
            friend class hpx::serialization::access;

            template <typename Archive>
            HPX_FORCEINLINE void serialize(Archive& ar, unsigned int const)
            {
                // clang-format off
                ar & bound_;
                // clang-format on
            }

            bound_type bound_;
        };
    }    // namespace detail

    template <typename Bound>
    functional::detail::post_continuation_impl<Bound, hpx::util::unused_type>
    post_continuation(Bound&& bound)
    {
        return functional::detail::post_continuation_impl<Bound,
            hpx::util::unused_type>(HPX_FORWARD(Bound, bound));
    }

    template <typename Bound, typename Continuation>
    functional::detail::post_continuation_impl<Bound, Continuation>
    post_continuation(Bound&& bound, Continuation&& c)
    {
        return functional::detail::post_continuation_impl<Bound, Continuation>(
            HPX_FORWARD(Bound, bound), HPX_FORWARD(Continuation, c));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Bound, typename Continuation>
        struct async_continuation_impl
        {
            using bound_type = typename std::decay<Bound>::type;
            using continuation_type = typename std::decay<Continuation>::type;

            async_continuation_impl() = default;

            template <typename Bound_, typename Continuation_>
            explicit async_continuation_impl(Bound_&& bound, Continuation_&& c)
              : bound_(HPX_FORWARD(Bound_, bound))
              , cont_(HPX_FORWARD(Continuation_, c))
            {
            }

            async_continuation_impl(async_continuation_impl&& o)
              : bound_(HPX_MOVE(o.bound_))
              , cont_(HPX_MOVE(o.cont_))
            {
            }

            async_continuation_impl& operator=(async_continuation_impl&& o)
            {
                bound_ = HPX_MOVE(o.bound_);
                cont_ = HPX_MOVE(o.cont_);
                return *this;
            }

            template <typename T>
            typename util::invoke_result<bound_type, hpx::id_type, T>::type
            operator()(hpx::id_type lco, T&& t)
            {
                typedef typename util::invoke_result<bound_type, hpx::id_type,
                    T>::type result_type;

                bound_.post_c(HPX_MOVE(cont_), lco, HPX_FORWARD(T, t));
                return result_type();
            }

        private:
            // serialization support
            friend class hpx::serialization::access;

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const)
            {
                // clang-format off
                ar & bound_ & cont_;
                // clang-format on
            }

            bound_type bound_;
            continuation_type cont_;
        };

        template <typename Bound>
        struct async_continuation_impl<Bound, hpx::util::unused_type>
        {
            using bound_type = typename std::decay<Bound>::type;

            async_continuation_impl() = default;

            template <typename Bound_,
                typename Enable = typename std::enable_if<
                    !std::is_same<typename std::decay<Bound_>::type,
                        async_continuation_impl>::value>::type>
            explicit async_continuation_impl(Bound_&& bound)
              : bound_(HPX_FORWARD(Bound_, bound))
            {
            }

            async_continuation_impl(async_continuation_impl&& o)
              : bound_(HPX_MOVE(o.bound_))
            {
            }

            async_continuation_impl& operator=(async_continuation_impl&& o)
            {
                bound_ = HPX_MOVE(o.bound_);
                return *this;
            }

            template <typename T>
            typename util::invoke_result<bound_type, hpx::id_type, T>::type
            operator()(hpx::id_type lco, T&& t)
            {
                typedef typename util::invoke_result<bound_type, hpx::id_type,
                    T>::type result_type;

                bound_.post_c(lco, lco, HPX_FORWARD(T, t));
                return result_type();
            }

        private:
            // serialization support
            friend class hpx::serialization::access;

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const)
            {
                // clang-format off
                ar & bound_;
                // clang-format on
            }

            bound_type bound_;
        };
    }    // namespace detail

    template <typename Bound>
    functional::detail::async_continuation_impl<Bound, hpx::util::unused_type>
    async_continuation(Bound&& bound)
    {
        return functional::detail::async_continuation_impl<Bound,
            hpx::util::unused_type>(HPX_FORWARD(Bound, bound));
    }

    template <typename Bound, typename Continuation>
    functional::detail::async_continuation_impl<Bound, Continuation>
    async_continuation(Bound&& bound, Continuation&& c)
    {
        return functional::detail::async_continuation_impl<Bound, Continuation>(
            HPX_FORWARD(Bound, bound), HPX_FORWARD(Continuation, c));
    }
}}}    // namespace hpx::util::functional
