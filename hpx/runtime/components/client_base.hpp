//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/action_remote_result.hpp>
#include <hpx/assert.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/acquire_future.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/make_client.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/traits/is_client.hpp>
#include <hpx/type_support/always_void.hpp>

#include <exception>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Client objects are equivalent to futures
namespace hpx { namespace components
{
    template <typename Derived, typename Stub>
    class client_base;
}}

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived>
    struct is_client<Derived,
            typename util::always_void<typename Derived::is_client_tag>::type>
      : std::true_type
    {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Derived>
        struct is_future_customization_point<Derived,
                typename std::enable_if<is_client<Derived>::value>::type>
          : std::true_type
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Derived>
        struct future_traits_customization_point<Derived,
            typename std::enable_if<is_client<Derived>::value>::type>
        {
            typedef id_type type;
            typedef id_type result_type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Derived>
        struct future_access_customization_point<Derived,
            typename std::enable_if<is_client<Derived>::value>::type>
        {
            template <typename SharedState>
            HPX_FORCEINLINE static Derived
            create(hpx::intrusive_ptr<SharedState> const& shared_state)
            {
                return Derived(future<id_type>(shared_state));
            }

            template <typename SharedState>
            HPX_FORCEINLINE static Derived
            create(hpx::intrusive_ptr<SharedState> && shared_state)
            {
                return Derived(future<id_type>(std::move(shared_state)));
            }

            template <typename SharedState>
            HPX_FORCEINLINE static Derived
            create(SharedState* shared_state, bool addref = true)
            {
                return Derived(future<id_type>(
                    hpx::intrusive_ptr<SharedState>(shared_state, addref)));
            }

            HPX_FORCEINLINE static
            typename traits::detail::shared_state_ptr<id_type>::type const&
            get_shared_state(Derived const& client)
            {
                return client.shared_state_;
            }

            HPX_FORCEINLINE static
            typename traits::detail::shared_state_ptr<id_type>::type::element_type*
            detach_shared_state(Derived const& f)
            {
                return f.shared_state_.get();
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Derived>
        struct acquire_future_impl<Derived,
            typename std::enable_if<is_client<Derived>::value>::type>
        {
            typedef Derived type;

            template <typename T_>
            HPX_FORCEINLINE
            Derived operator()(T_ && value) const
            {
                return std::forward<T_>(value);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Derived>
        struct shared_state_ptr_for<Derived,
                typename std::enable_if<is_client<Derived>::value>::type>
          : shared_state_ptr<typename traits::future_traits<Derived>::type>
        {};
    }
}}

namespace hpx { namespace lcos { namespace detail
{
    template <typename Derived>
    struct future_unwrap_result<Derived,
        typename std::enable_if<traits::is_client<Derived>::value>::type>
    {
        typedef id_type result_type;
        typedef Derived type;
    };

    template <typename Derived>
    struct future_unwrap_result<future<Derived>,
        typename std::enable_if<traits::is_client<Derived>::value>::type>
    {
        typedef id_type result_type;
        typedef Derived type;
    };
}}}

namespace hpx { namespace lcos { namespace detail
{
    // Specialization for shared state of id_type, additionally (optionally)
    // holds a registered name for the object it refers to.
    template <>
    struct future_data<id_type> : future_data_base<id_type>
    {
        HPX_NON_COPYABLE(future_data);

        using init_no_addref =
            typename future_data_base<id_type>::init_no_addref;

        future_data() = default;

        future_data(init_no_addref no_addref)
          : future_data_base<id_type>(no_addref)
        {}

        template <typename ... T>
        future_data(init_no_addref no_addref, in_place in_place, T&& ... ts)
          : future_data_base<id_type>(no_addref, in_place, std::forward<T>(ts)...)
        {}

        future_data(init_no_addref no_addref, std::exception_ptr const& e)
          : future_data_base<id_type>(no_addref, e)
        {}
        future_data(init_no_addref no_addref, std::exception_ptr && e)
          : future_data_base<id_type>(no_addref, std::move(e))
        {}

        ~future_data() noexcept override
        {
            if (!registered_name_.empty())
            {
                std::string name = std::move(registered_name_);
                error_code ec(lightweight);
                agas::unregister_name(launch::sync, name, ec);
            }
        }

        std::string const& get_registered_name() const override
        {
            return registered_name_;
        }
        void register_as(std::string const& name, bool manage_lifetime) override
        {
            HPX_ASSERT(registered_name_.empty());   // call only once
            registered_name_ = name;
            if (manage_lifetime)
            {
                hpx::agas::register_name(launch::sync, name, *this->get_result());
            }
        }

    private:
        std::string registered_name_;
    };
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    namespace detail
    {
        // Wrap a given type such that it is usable as a stub_base.
        // The main template is chosen whenever the argument Stub is not a (or
        // not derived from) a stub_base. In this case Stub should be a server
        // side component implementation type.
        template <typename Stub, typename Enable = void>
        struct make_stub
        {
            typedef components::stub_base<Stub> type;
            typedef typename components::stub_base<Stub>::server_component_type
                server_component_type;
        };

        // This specialization is chosen whenever the argument Stub is a (or
        // derived from a) stub_base.
        template <typename Stub>
        struct make_stub<Stub, typename util::always_void<
            typename Stub::server_component_type>::type>
        {
            typedef Stub type;
            typedef typename Stub::server_component_type server_component_type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename Stub>
    class client_base : public detail::make_stub<Stub>::type
    {
    private:
        template <typename T, typename Enable>
        friend struct hpx::traits::detail::future_access_customization_point;

    protected:
        typedef typename detail::make_stub<Stub>::type stub_type;
        typedef lcos::detail::future_data_base<id_type> shared_state_type;

        typedef shared_future<id_type> future_type;

        client_base(hpx::intrusive_ptr<shared_state_type> const& state)
          : shared_state_(state)
        {}

        client_base(hpx::intrusive_ptr<shared_state_type> && state)
          : shared_state_(std::move(state))
        {}

    public:
        typedef Stub stub_argument_type;
        typedef typename detail::make_stub<Stub>::server_component_type
            server_component_type;

        typedef void is_client_tag;

        client_base()
          : shared_state_()
        {}

        explicit client_base(id_type const& id)
          : shared_state_(new lcos::detail::future_data<id_type>)
        {
            shared_state_->set_value(id);
        }
        explicit client_base(id_type && id)
          : shared_state_(new lcos::detail::future_data<id_type>)
        {
            shared_state_->set_value(std::move(id));
        }

        explicit client_base(shared_future<id_type> const& f) noexcept
          : shared_state_(
                hpx::traits::future_access<future_type>::
                    get_shared_state(f))
        {}
        explicit client_base(shared_future<id_type> && f) noexcept
          : shared_state_(
                hpx::traits::future_access<future_type>::
                    get_shared_state(std::move(f)))
        {}
        explicit client_base(future<id_type> && f) noexcept
          : shared_state_(hpx::traits::future_access<future_type>::
                    get_shared_state(std::move(f)))
        {}

        client_base(client_base const& rhs) noexcept
          : shared_state_(rhs.shared_state_)
        {}
        client_base(client_base && rhs) noexcept
          : shared_state_(std::move(rhs.shared_state_))
        {
            rhs.shared_state_ = nullptr;
        }

        // A future to a client_base can be unwrap to represent the
        // client_base directly as a client_base is semantically a future to
        // the id of the referenced object.
        client_base(future<Derived> && d)
          : shared_state_(d.valid() ? lcos::detail::unwrap(std::move(d)) : nullptr)
        {}

        ~client_base() = default;

        // copy assignment and move assignment
        client_base& operator=(id_type const& id)
        {
            shared_state_ = new shared_state_type;
            shared_state_->set_value(id);
            return *this;
        }
        client_base& operator=(id_type && id)
        {
            shared_state_ = new shared_state_type;
            shared_state_->set_value(std::move(id));
            return *this;
        }

        client_base& operator=(shared_future<id_type> const& f)
        {
            shared_state_ = hpx::traits::future_access<future_type>::
                get_shared_state(f);
            return *this;
        }
        client_base& operator=(shared_future<id_type> && f)
        {
            shared_state_ = hpx::traits::future_access<future_type>::
                get_shared_state(std::move(f));
            return *this;
        }
        client_base& operator=(future<id_type> && f)
        {
            shared_state_ = hpx::traits::future_access<future_type>::
                get_shared_state(std::move(f));
            return *this;
        }

        client_base& operator=(client_base const& rhs)
        {
            shared_state_ = rhs.shared_state_;
            return *this;
        }
        client_base& operator=(client_base && rhs)
        {
            shared_state_ = std::move(rhs.shared_state_);
            return *this;
        }

        // Returns: true only if *this refers to a shared state.
        bool valid() const noexcept
        {
            return shared_state_ != nullptr;
        }

        // check whether the embedded shared state is valid
        explicit operator bool() const noexcept
        {
            return valid();
        }

        void free()
        {
            shared_state_.reset();
        }

        ///////////////////////////////////////////////////////////////////////
        id_type const& get_id() const
        {
            return get();
        }

        naming::gid_type const& get_raw_gid() const
        {
            return get_id().get_gid();
        }

        ///////////////////////////////////////////////////////////////////////
        shared_future<id_type> detach()
        {
            return hpx::traits::future_access<future_type>::
                create(std::move(shared_state_));
        }

        shared_future<id_type> share() const
        {
            return hpx::traits::future_access<future_type>::
                create(shared_state_);
        }

        void reset(id_type const& id)
        {
            *this = id;
        }

        void reset(id_type && id)
        {
            *this = std::move(id);
        }

        void reset(shared_future<id_type> && rhs)
        {
            *this = std::move(rhs);
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        // Exposition only: interface mimicking future

        id_type const& get() const
        {
            if (!shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "client_base::get_gid",
                    "this client_base has no valid shared state");
            }

            // no error has been reported, return the result
            return lcos::detail::future_value<id_type>::
                get(*shared_state_->get_result());
        }

        // Returns: true if the shared state is ready, false if it isn't.
        bool is_ready() const noexcept
        {
            return shared_state_ != nullptr && shared_state_->is_ready();
        }

        // Returns: true if the shared state is ready and stores a value,
        //          false if it isn't.
        bool has_value() const noexcept
        {
            return shared_state_ != nullptr && shared_state_->has_value();
        }

        // Returns: true if the shared state is ready and stores an exception,
        //          false if it isn't.
        bool has_exception() const noexcept
        {
            return shared_state_ != nullptr && shared_state_->has_exception();
        }

        void wait() const
        {
            if (!shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "client_base::wait",
                    "this client_base has no valid shared state");
                return;
            }
            shared_state_->wait();
        }

        // Effects:
        //   - Blocks until the future is ready.
        // Returns: The stored exception_ptr if has_exception(), a null
        //          pointer otherwise.
        std::exception_ptr get_exception_ptr() const
        {
            if (!shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "client_base<Derived, Stub>::get_exception_ptr",
                    "this client has no valid shared state");
            }

            error_code ec(lightweight);
            this->shared_state_->get_result(ec);
            if (!ec) return std::exception_ptr();
            return hpx::detail::access_exception(ec);
        }

    private:
        template <typename F>
        static typename hpx::traits::future_then_result<Derived, F>::cont_result
        on_ready(shared_future<id_type> && fut, F f)
        {
            return f(Derived(std::move(fut)));
        }

    public:
        template <typename F>
        typename hpx::traits::future_then_result<Derived, F>::type
        then(F && f)
        {
            typedef
                typename hpx::traits::future_then_result<Derived, F>::result_type
                result_type;

            if (!shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "client_base::then",
                    "this client_base has no valid shared state");
                return future<result_type>();
            }

            typedef
                typename hpx::util::invoke_result<F, Derived>::type
                continuation_result_type;
            typedef
                typename hpx::traits::detail::shared_state_ptr<result_type>::type
                shared_state_ptr;

            shared_state_ptr p =
                lcos::detail::make_continuation<continuation_result_type>(
                    *static_cast<Derived const*>(this), launch::all,
                    std::forward<F>(f));
            return hpx::traits::future_access<future<result_type> >::create(
                std::move(p));
        }

    private:
        ///////////////////////////////////////////////////////////////////////
        static void register_as_helper(client_base const& f,
            std::string const& symbolic_name, bool manage_lifetime)
        {
            f.shared_state_->register_as(symbolic_name, manage_lifetime);
        }

    public:
        // Register our id with AGAS using the given name
        future<void> register_as(std::string const& symbolic_name,
            bool manage_lifetime = true)
        {
            if (!shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "client_base::register_as",
                    "this client_base has no valid shared state");
            }

            typename hpx::traits::detail::shared_state_ptr<void>::type p =
                lcos::detail::make_continuation<void>(
                    *this, launch::sync,
                    [=](client_base const& f) -> void {
                        return register_as_helper(
                            f, symbolic_name, manage_lifetime);
                    });

            return hpx::traits::future_access<future<void> >::
                create(std::move(p));
        }

        // Retrieve the id associated with the given name and use it to
        // initialize this client_base instance.
        void connect_to(std::string const& symbolic_name)
        {
            *this = agas::on_symbol_namespace_event(symbolic_name, true);
        }

        // Make sure this instance does not manage the lifetime of the
        // registered object anymore (obsolete).
        HPX_DEPRECATED_V(1, 4, HPX_DEPRECATED_MSG) void reset_registered_name()
        {
        }

        // Access registered name for the component
        std::string const& registered_name() const
        {
            return shared_state_->get_registered_name();
        }

    protected:
        // shared state holding the id_type this client refers to
        hpx::intrusive_ptr<shared_state_type> shared_state_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename Stub>
    bool operator==(
        client_base<Derived, Stub> const& lhs,
        client_base<Derived, Stub> const& rhs)
    {
        return lhs.get() == rhs.get();
    }

    template <typename Derived, typename Stub>
    bool operator<(
        client_base<Derived, Stub> const& lhs,
        client_base<Derived, Stub> const& rhs)
    {
        return lhs.get() < rhs.get();
    }
}}

namespace hpx { namespace serialization
{
    template <typename Archive, typename Derived, typename Stub>
    HPX_FORCEINLINE
    void serialize(Archive& ar,
        ::hpx::components::client_base<Derived, Stub>& f, unsigned version)
    {
        hpx::lcos::detail::serialize_future(ar, f, version);
    }
}}


