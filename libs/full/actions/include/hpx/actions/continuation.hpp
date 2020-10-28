//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/continuation_fwd.hpp>
#include <hpx/actions_base/action_priority.hpp>
#include <hpx/actions_base/basic_action_fwd.hpp>
#include <hpx/actions_base/traits/action_remote_result.hpp>
#include <hpx/functional/serialization/serializable_unique_function.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/trigger_lco_fwd.hpp>
#include <hpx/serialization/base_object.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/traits/is_continuation.hpp>

#include <exception>
#include <type_traits>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions {

    ///////////////////////////////////////////////////////////////////////////
    // Continuations are polymorphic objects encapsulating the
    // id_type of the destination where the result has to be sent.
    class HPX_EXPORT continuation
    {
    public:
        typedef void continuation_tag;

        continuation();

        explicit continuation(naming::id_type const& id);
        explicit continuation(naming::id_type&& id);

        continuation(naming::id_type const& id, naming::address&& addr);
        continuation(naming::id_type&& id, naming::address&& addr) noexcept;

        continuation(continuation&& o) noexcept;
        continuation& operator=(continuation&& o) noexcept;

        //
        void trigger_error(std::exception_ptr const& e);
        void trigger_error(std::exception_ptr&& e);

        // serialization support
        void serialize(hpx::serialization::input_archive& ar, unsigned);
        void serialize(hpx::serialization::output_archive& ar, unsigned);

        constexpr naming::id_type const& get_id() const noexcept
        {
            return id_;
        }

        constexpr naming::address get_addr() const noexcept
        {
            return addr_;
        }

    protected:
        naming::id_type id_;
        naming::address addr_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct typed_continuation<Result, Result> : continuation
    {
    private:
        using function_type =
            util::unique_function<void(naming::id_type, Result)>;

    public:
        using result_type = Result;

        typed_continuation() = default;

        explicit typed_continuation(naming::id_type const& id)
          : continuation(id)
        {
        }

        explicit typed_continuation(naming::id_type&& id) noexcept
          : continuation(std::move(id))
        {
        }

        template <typename F>
        typed_continuation(naming::id_type const& id, F&& f)
          : continuation(id)
          , f_(std::forward<F>(f))
        {
        }

        template <typename F>
        typed_continuation(naming::id_type&& id, F&& f)
          : continuation(std::move(id))
          , f_(std::forward<F>(f))
        {
        }

        typed_continuation(naming::id_type const& id, naming::address&& addr)
          : continuation(id, std::move(addr))
        {
        }

        typed_continuation(
            naming::id_type&& id, naming::address&& addr) noexcept
          : continuation(std::move(id), std::move(addr))
        {
        }

        template <typename F>
        typed_continuation(
            naming::id_type const& id, naming::address&& addr, F&& f)
          : continuation(id, std::move(addr))
          , f_(std::forward<F>(f))
        {
        }

        template <typename F>
        typed_continuation(naming::id_type&& id, naming::address&& addr, F&& f)
          : continuation(std::move(id), std::move(addr))
          , f_(std::forward<F>(f))
        {
        }

        template <typename F,
            typename Enable = typename std::enable_if<!std::is_same<
                typename std::decay<F>::type, typed_continuation>::value>::type>
        explicit typed_continuation(F&& f)
          : f_(std::forward<F>(f))
        {
        }

        typed_continuation(typed_continuation&&) noexcept = default;
        typed_continuation& operator=(typed_continuation&&) noexcept = default;

        void trigger_value(Result&& result)
        {
            LLCO_(info) << "typed_continuation<Result>::trigger_value("
                        << this->get_id() << ")";

            if (f_.empty())
            {
                if (!this->get_id())
                {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<Result>::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                hpx::set_lco_value(
                    this->get_id(), this->get_addr(), std::move(result));
            }
            else
            {
                f_(this->get_id(), std::move(result));
            }
        }

    private:
        /// serialization support
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            // clang-format off
            // serialize base class
            ar & hpx::serialization::base_object<continuation>(*this);
            ar & f_;
            // clang-format on
        }

    protected:
        function_type f_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // This specialization is needed to call the right
    // base_lco_with_value action if the local Result is computed
    // via get_remote_result and differs from the actions original
    // local result type
    template <typename Result, typename RemoteResult>
    struct typed_continuation : typed_continuation<RemoteResult>
    {
    private:
        using base_type = typed_continuation<RemoteResult>;
        using function_type =
            util::unique_function<void(naming::id_type, RemoteResult)>;

    public:
        typed_continuation() = default;

        explicit typed_continuation(naming::id_type const& id)
          : base_type(id)
        {
        }

        explicit typed_continuation(naming::id_type&& id) noexcept
          : base_type(std::move(id))
        {
        }

        template <typename F>
        typed_continuation(naming::id_type const& id, F&& f)
          : base_type(id, std::forward<F>(f))
        {
        }

        template <typename F>
        typed_continuation(naming::id_type&& id, F&& f)
          : base_type(std::move(id), std::forward<F>(f))
        {
        }

        typed_continuation(naming::id_type const& id, naming::address&& addr)
          : base_type(id, std::move(addr))
        {
        }

        typed_continuation(
            naming::id_type&& id, naming::address&& addr) noexcept
          : base_type(std::move(id), std::move(addr))
        {
        }

        template <typename F>
        typed_continuation(
            naming::id_type const& id, naming::address&& addr, F&& f)
          : base_type(id, std::move(addr), std::forward<F>(f))
        {
        }

        template <typename F>
        typed_continuation(naming::id_type&& id, naming::address&& addr, F&& f)
          : base_type(std::move(id), std::move(addr), std::forward<F>(f))
        {
        }

        template <typename F,
            typename Enable = typename std::enable_if<!std::is_same<
                typename std::decay<F>::type, typed_continuation>::value>::type>
        explicit typed_continuation(F&& f)
          : base_type(std::forward<F>(f))
        {
        }

        typed_continuation(typed_continuation&&) noexcept = default;
        typed_continuation& operator=(typed_continuation&&) noexcept = default;

        void trigger_value(RemoteResult&& result)
        {
            LLCO_(info) << "typed_continuation<RemoteResult>::trigger_value("
                        << this->get_id() << ")";

            if (this->f_.empty())
            {
                if (!this->get_id())
                {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<Result>::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                hpx::set_lco_value(
                    this->get_id(), this->get_addr(), std::move(result));
            }
            else
            {
                this->f_(this->get_id(), std::move(result));
            }
        }

    private:
        /// serialization support
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            // clang-format off
            // serialize base class
            ar & hpx::serialization::base_object<
                typed_continuation<RemoteResult> >(*this);
            // clang-format on
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    struct typed_continuation<void, util::unused_type> : continuation
    {
    private:
        using function_type = util::unique_function<void(naming::id_type)>;

    public:
        using result_type = void;

        typed_continuation() = default;

        explicit typed_continuation(naming::id_type const& id)
          : continuation(id)
        {
        }

        explicit typed_continuation(naming::id_type&& id) noexcept
          : continuation(std::move(id))
        {
        }

        template <typename F>
        typed_continuation(naming::id_type const& id, F&& f)
          : continuation(id)
          , f_(std::forward<F>(f))
        {
        }

        template <typename F>
        typed_continuation(naming::id_type&& id, F&& f)
          : continuation(std::move(id))
          , f_(std::forward<F>(f))
        {
        }

        typed_continuation(naming::id_type const& id, naming::address&& addr)
          : continuation(id, std::move(addr))
        {
        }

        typed_continuation(
            naming::id_type&& id, naming::address&& addr) noexcept
          : continuation(std::move(id), std::move(addr))
        {
        }

        template <typename F>
        typed_continuation(
            naming::id_type const& id, naming::address&& addr, F&& f)
          : continuation(id, std::move(addr))
          , f_(std::forward<F>(f))
        {
        }

        template <typename F>
        typed_continuation(naming::id_type&& id, naming::address&& addr, F&& f)
          : continuation(std::move(id), std::move(addr))
          , f_(std::forward<F>(f))
        {
        }

        template <typename F,
            typename Enable = typename std::enable_if<!std::is_same<
                typename std::decay<F>::type, typed_continuation>::value>::type>
        explicit typed_continuation(F&& f)
          : f_(std::forward<F>(f))
        {
        }

        typed_continuation(typed_continuation&&) noexcept = default;
        typed_continuation& operator=(typed_continuation&&) noexcept = default;

        HPX_EXPORT void trigger();

        void trigger_value(util::unused_type&&)
        {
            this->trigger();
        }

        void trigger_value(util::unused_type const&)
        {
            this->trigger();
        }

    private:
        /// serialization support
        friend class hpx::serialization::access;

        HPX_EXPORT void serialize(
            hpx::serialization::input_archive& ar, unsigned);
        HPX_EXPORT void serialize(
            hpx::serialization::output_archive& ar, unsigned);

        function_type f_;
    };
}}    // namespace hpx::actions

#include <hpx/config/warnings_suffix.hpp>

// this file is intentionally #included last as it refers to functions defined
// here
#include <hpx/runtime/trigger_lco.hpp>
