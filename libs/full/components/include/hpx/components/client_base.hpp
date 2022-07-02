//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/action_remote_result.hpp>
#include <hpx/actions_base/traits/is_client.hpp>
#include <hpx/assert.hpp>
#include <hpx/components/basename_registration.hpp>
#include <hpx/components/components_fwd.hpp>
#include <hpx/components/make_client.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/stub_base.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/acquire_future.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/naming_base/unmanaged.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/type_support/always_void.hpp>

#include <exception>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Client objects are equivalent to futures
namespace hpx { namespace lcos { namespace detail {

    // Specialization for shared state of id_type, additionally (optionally)
    // holds a registered name for the object it refers to.
    template <>
    struct future_data<id_type>;
}}}    // namespace hpx::lcos::detail

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived>
    struct is_client<Derived,
        typename util::always_void<typename Derived::is_client_tag>::type>
      : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename Derived>
        struct is_future_customization_point<Derived,
            std::enable_if_t<is_client_v<Derived>>> : std::true_type
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Derived>
        struct future_traits_customization_point<Derived,
            std::enable_if_t<is_client_v<Derived>>>
        {
            using type = id_type;
            using result_type = id_type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Derived>
        struct future_access_customization_point<Derived,
            std::enable_if_t<is_client_v<Derived>>>
        {
            template <typename SharedState>
            HPX_FORCEINLINE static Derived create(
                hpx::intrusive_ptr<SharedState> const& shared_state)
            {
                return Derived(hpx::future<id_type>(shared_state));
            }

            template <typename SharedState>
            HPX_FORCEINLINE static Derived create(
                hpx::intrusive_ptr<SharedState>&& shared_state)
            {
                return Derived(hpx::future<id_type>(HPX_MOVE(shared_state)));
            }

            template <typename SharedState>
            HPX_FORCEINLINE static Derived create(
                SharedState* shared_state, bool addref = true)
            {
                return Derived(hpx::future<id_type>(
                    hpx::intrusive_ptr<SharedState>(shared_state, addref)));
            }

            HPX_FORCEINLINE static traits::detail::shared_state_ptr_t<
                id_type> const&
            get_shared_state(Derived const& client)
            {
                return client.shared_state_;
            }

            HPX_FORCEINLINE static typename traits::detail::shared_state_ptr_t<
                id_type>::element_type*
            detach_shared_state(Derived const& f)
            {
                return f.shared_state_.get();
            }

            template <typename Destination>
            HPX_FORCEINLINE static void transfer_result(
                Derived&& src, Destination& dest)
            {
                dest.set_value(src.get());
                dest.set_registered_name(
                    src.shared_state_->get_registered_name());
                src.shared_state_->set_registered_name(std::string());
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Derived>
        struct acquire_future_impl<Derived,
            std::enable_if_t<is_client_v<Derived>>>
        {
            using type = Derived;

            template <typename T_>
            HPX_FORCEINLINE Derived operator()(T_&& value) const
            {
                return HPX_FORWARD(T_, value);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Derived>
        struct shared_state_ptr_for<Derived,
            std::enable_if_t<is_client_v<Derived>>>
          : shared_state_ptr<traits::future_traits_t<Derived>>
        {
        };
    }    // namespace detail
}}       // namespace hpx::traits

namespace hpx { namespace lcos { namespace detail {

    template <typename Derived>
    struct future_unwrap_result<Derived,
        std::enable_if_t<traits::is_client_v<Derived>>>
    {
        using type = id_type;
        using wrapped_type = Derived;
    };

    template <typename Derived>
    struct future_unwrap_result<hpx::future<Derived>,
        std::enable_if_t<traits::is_client_v<Derived>>>
    {
        using type = id_type;
        using wrapped_type = Derived;
    };

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
        {
        }

        template <typename... T>
        future_data(init_no_addref no_addref, in_place in_place, T&&... ts)
          : future_data_base<id_type>(
                no_addref, in_place, HPX_FORWARD(T, ts)...)
        {
        }

        future_data(init_no_addref no_addref, std::exception_ptr const& e)
          : future_data_base<id_type>(no_addref, e)
        {
        }
        future_data(init_no_addref no_addref, std::exception_ptr&& e)
          : future_data_base<id_type>(no_addref, HPX_MOVE(e))
        {
        }

        ~future_data() noexcept override
        {
            if (!registered_name_.empty())
            {
                std::string name = HPX_MOVE(registered_name_);
                error_code ec(throwmode::lightweight);
                agas::unregister_name(launch::sync, name, ec);
            }
        }

        std::string const& get_registered_name() const override
        {
            return registered_name_;
        }
        void set_registered_name(std::string name) override
        {
            registered_name_ = HPX_MOVE(name);
        }
        bool register_as(std::string name, bool manage_lifetime) override
        {
            HPX_ASSERT(registered_name_.empty());    // call only once
            registered_name_ = HPX_MOVE(name);
            hpx::id_type id = *this->get_result();
            if (!manage_lifetime)
            {
                id = hpx::naming::unmanaged(id);
            }
            return hpx::agas::register_name(launch::sync, registered_name_, id);
        }

        std::string registered_name_;
    };
}}}    // namespace hpx::lcos::detail

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components {

    namespace detail {
        // Wrap a given type such that it is usable as a stub_base.
        // The main template is chosen whenever the argument Stub is not a (or
        // not derived from) a stub_base. In this case Stub should be a server
        // side component implementation type.
        template <typename Stub, typename Enable = void>
        struct make_stub
        {
            using type = components::stub_base<Stub>;
            using server_component_type =
                typename components::stub_base<Stub>::server_component_type;
        };

        // This specialization is chosen whenever the argument Stub is a (or
        // derived from a) stub_base.
        template <typename Stub>
        struct make_stub<Stub,
            util::always_void_t<typename Stub::server_component_type>>
        {
            using type = Stub;
            using server_component_type = typename Stub::server_component_type;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename Stub>
    class client_base : public detail::make_stub<Stub>::type
    {
    private:
        template <typename T, typename Enable>
        friend struct hpx::traits::detail::future_access_customization_point;

    protected:
        using stub_type = typename detail::make_stub<Stub>::type;
        using shared_state_type = lcos::detail::future_data_base<id_type>;

        using future_type = shared_future<id_type>;

        client_base(hpx::intrusive_ptr<shared_state_type> const& state)
          : shared_state_(state)
        {
        }

        client_base(hpx::intrusive_ptr<shared_state_type>&& state)
          : shared_state_(HPX_MOVE(state))
        {
        }

    public:
        using stub_argument_type = Stub;
        using server_component_type =
            typename detail::make_stub<Stub>::server_component_type;

        using is_client_tag = void;

        client_base() = default;

        explicit client_base(id_type const& id)
          : shared_state_(new lcos::detail::future_data<id_type>)
        {
            shared_state_->set_value(id);
        }
        explicit client_base(id_type&& id)
          : shared_state_(new lcos::detail::future_data<id_type>)
        {
            shared_state_->set_value(HPX_MOVE(id));
        }

        explicit client_base(hpx::shared_future<id_type> const& f) noexcept
          : shared_state_(
                hpx::traits::future_access<future_type>::get_shared_state(f))
        {
        }
        explicit client_base(hpx::shared_future<id_type>&& f) noexcept
          : shared_state_(
                hpx::traits::future_access<future_type>::get_shared_state(
                    HPX_MOVE(f)))
        {
        }
        explicit client_base(hpx::future<id_type>&& f) noexcept
          : shared_state_(
                hpx::traits::future_access<future_type>::get_shared_state(
                    HPX_MOVE(f)))
        {
        }

        client_base(client_base const& rhs) noexcept
          : shared_state_(rhs.shared_state_)
        {
        }
        client_base(client_base&& rhs) noexcept
          : shared_state_(HPX_MOVE(rhs.shared_state_))
        {
            rhs.shared_state_ = nullptr;
        }

        // A future to a client_base can be unwrap to represent the
        // client_base directly as a client_base is semantically a future to
        // the id of the referenced object.
        client_base(hpx::future<Derived>&& d)
          : shared_state_(
                d.valid() ? lcos::detail::unwrap(HPX_MOVE(d)) : nullptr)
        {
        }

        ~client_base() = default;

        // copy assignment and move assignment
        client_base& operator=(id_type const& id)
        {
            shared_state_ = new shared_state_type;
            shared_state_->set_value(id);
            return *this;
        }
        client_base& operator=(id_type&& id)
        {
            shared_state_ = new shared_state_type;
            shared_state_->set_value(HPX_MOVE(id));
            return *this;
        }

        client_base& operator=(hpx::shared_future<id_type> const& f) noexcept
        {
            shared_state_ =
                hpx::traits::future_access<future_type>::get_shared_state(f);
            return *this;
        }
        client_base& operator=(hpx::shared_future<id_type>&& f) noexcept
        {
            shared_state_ =
                hpx::traits::future_access<future_type>::get_shared_state(
                    HPX_MOVE(f));
            return *this;
        }
        client_base& operator=(hpx::future<id_type>&& f) noexcept
        {
            shared_state_ =
                hpx::traits::future_access<future_type>::get_shared_state(
                    HPX_MOVE(f));
            return *this;
        }

        client_base& operator=(client_base const& rhs) noexcept
        {
            shared_state_ = rhs.shared_state_;
            return *this;
        }
        client_base& operator=(client_base&& rhs) noexcept
        {
            shared_state_ = HPX_MOVE(rhs.shared_state_);
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
        hpx::shared_future<id_type> detach()
        {
            return hpx::traits::future_access<future_type>::create(
                HPX_MOVE(shared_state_));
        }

        hpx::shared_future<id_type> share() const
        {
            return hpx::traits::future_access<future_type>::create(
                shared_state_);
        }

        void reset(id_type const& id)
        {
            *this = id;
        }

        void reset(id_type&& id)
        {
            *this = HPX_MOVE(id);
        }

        void reset(shared_future<id_type>&& rhs)
        {
            *this = HPX_MOVE(rhs);
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        // Exposition only: interface mimicking future

        id_type const& get() const
        {
            if (!shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state, "client_base::get_gid",
                    "this client_base has no valid shared state");
            }

            // no error has been reported, return the result
            return lcos::detail::future_value<id_type>::get(
                *shared_state_->get_result());
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
                HPX_THROW_EXCEPTION(no_state, "client_base::wait",
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

            error_code ec(throwmode::lightweight);
            this->shared_state_->get_result(ec);
            if (!ec)
                return std::exception_ptr();
            return hpx::detail::access_exception(ec);
        }

    private:
        template <typename F>
        static typename hpx::traits::future_then_result<Derived, F>::cont_result
        on_ready(hpx::shared_future<id_type>&& fut, F f)
        {
            return f(Derived(HPX_MOVE(fut)));
        }

    public:
        template <typename F>
        hpx::traits::future_then_result_t<Derived, F> then(launch l, F&& f)
        {
            using result_type =
                typename hpx::traits::future_then_result<Derived,
                    F>::result_type;

            if (!shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state, "client_base::then",
                    "this client_base has no valid shared state");
                return hpx::future<result_type>();
            }

            using continuation_result_type =
                hpx::util::invoke_result_t<F, Derived>;
            using shared_state_ptr =
                hpx::traits::detail::shared_state_ptr_t<result_type>;

            shared_state_ptr p =
                lcos::detail::make_continuation<continuation_result_type>(
                    *static_cast<Derived const*>(this), l, HPX_FORWARD(F, f));
            return hpx::traits::future_access<hpx::future<result_type>>::create(
                HPX_MOVE(p));
        }

        template <typename F>
        hpx::traits::future_then_result_t<Derived, F> then(
            launch::sync_policy l, F&& f)
        {
            using func_result = decltype(HPX_FORWARD(F, f)(Derived(*this)));
            using future_result = hpx::traits::future_then_result_t<Derived, F>;
            if constexpr (std::is_convertible_v<func_result, future_result>)
            {
                if (is_ready())
                {
                    return HPX_FORWARD(F, f)(Derived(*this));
                }
            }
            return then(launch(l), HPX_FORWARD(F, f));
        }

        template <typename F>
        hpx::traits::future_then_result_t<Derived, F> then(F&& f)
        {
            return then(launch::all, HPX_FORWARD(F, f));
        }

    private:
        ///////////////////////////////////////////////////////////////////////
        static bool register_as_helper(client_base const& f,
            std::string symbolic_name, bool manage_lifetime)
        {
            return f.shared_state_->register_as(
                HPX_MOVE(symbolic_name), manage_lifetime);
        }

    public:
        // Register our id with AGAS using the given name
        hpx::future<bool> register_as(
            std::string symbolic_name, bool manage_lifetime = true)
        {
            if (!shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state, "client_base::register_as",
                    "this client_base has no valid shared state");
            }

            hpx::traits::detail::shared_state_ptr_t<bool> p =
                lcos::detail::make_continuation<bool>(*this, launch::sync,
                    [=, symbolic_name = HPX_MOVE(symbolic_name)](
                        client_base const& f) mutable -> bool {
                        return register_as_helper(
                            f, HPX_MOVE(symbolic_name), manage_lifetime);
                    });

            return hpx::traits::future_access<hpx::future<bool>>::create(
                HPX_MOVE(p));
        }

        // Retrieve the id associated with the given name and use it to
        // initialize this client_base instance.
        void connect_to(std::string const& symbolic_name)
        {
            *this = agas::on_symbol_namespace_event(symbolic_name, true);
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
    bool operator==(client_base<Derived, Stub> const& lhs,
        client_base<Derived, Stub> const& rhs)
    {
        return lhs.get() == rhs.get();
    }

    template <typename Derived, typename Stub>
    bool operator<(client_base<Derived, Stub> const& lhs,
        client_base<Derived, Stub> const& rhs)
    {
        return lhs.get() < rhs.get();
    }
}}    // namespace hpx::components

namespace hpx { namespace serialization {

    template <typename Archive, typename Derived, typename Stub>
    HPX_FORCEINLINE void serialize(Archive& ar,
        ::hpx::components::client_base<Derived, Stub>& f, unsigned version)
    {
        hpx::lcos::detail::serialize_future(ar, f, version);
    }
}}    // namespace hpx::serialization
