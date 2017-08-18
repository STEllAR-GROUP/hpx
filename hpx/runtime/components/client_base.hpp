//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CLIENT_BASE_OCT_31_2008_0424PM)
#define HPX_COMPONENTS_CLIENT_BASE_OCT_31_2008_0424PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/make_client.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/naming/unmanaged.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/traits/action_remote_result.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/is_client.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/util/always_void.hpp>

#include <boost/intrusive_ptr.hpp>

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
            create(boost::intrusive_ptr<SharedState> const& shared_state)
            {
                return Derived(future<id_type>(shared_state));
            }

            template <typename SharedState>
            HPX_FORCEINLINE static Derived
            create(boost::intrusive_ptr<SharedState> && shared_state)
            {
                return Derived(future<id_type>(std::move(shared_state)));
            }

            template <typename SharedState>
            HPX_FORCEINLINE static Derived
            create(SharedState* shared_state)
            {
                return Derived(future<id_type>(
                    boost::intrusive_ptr<SharedState>(shared_state)));
            }

            HPX_FORCEINLINE static
            typename traits::detail::shared_state_ptr<id_type>::type const&
            get_shared_state(Derived const& client)
            {
                return client.shared_state_;
            }

#if BOOST_VERSION >= 105600
            HPX_FORCEINLINE static
            typename traits::detail::shared_state_ptr<id_type>::type::element_type*
            detach_shared_state(Derived const& f)
            {
                return f.shared_state_.get();
            }
#endif
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

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    namespace detail
    {
        // Wrap a give type such that it is usable as a stub_base.
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
        typedef lcos::detail::future_data<id_type> shared_state_type;

        typedef shared_future<id_type> future_type;

        void unregister_held_object(error_code& ec = throws)
        {
            if (!registered_name_.empty())
            {
                std::string name = std::move(registered_name_);
                agas::unregister_name(launch::sync, name, ec);
            }
        }

        client_base(boost::intrusive_ptr<shared_state_type> const& state)
          : shared_state_(state)
        {}

        client_base(boost::intrusive_ptr<shared_state_type> && state)
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
          : shared_state_(new shared_state_type)
        {
            shared_state_->set_value(id);
        }
        explicit client_base(id_type && id)
          : shared_state_(new shared_state_type)
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
          : registered_name_(std::move(rhs.registered_name_)),
            shared_state_(std::move(rhs.shared_state_))
        {
            rhs.shared_state_ = nullptr;
        }

        // A future to a client_base can be unwrap to represent the
        // client_base directly as a client_base is semantically a future to
        // the id of the referenced object.
        client_base(future<Derived> && d)
          : shared_state_(d.valid() ? lcos::detail::unwrap(std::move(d)) : nullptr)
        {}

        ~client_base()
        {
            error_code ec;              // ignore all exceptions
            unregister_held_object(ec);
        }

        // copy assignment and move assignment
        client_base& operator=(id_type const& id)
        {
            unregister_held_object();
            shared_state_ = new shared_state_type;
            shared_state_->set_value(id);
            return *this;
        }
        client_base& operator=(id_type && id)
        {
            unregister_held_object();
            shared_state_ = new shared_state_type;
            shared_state_->set_value(std::move(id));
            return *this;
        }

        client_base& operator=(shared_future<id_type> const& f)
        {
            unregister_held_object();
            shared_state_ = hpx::traits::future_access<future_type>::
                get_shared_state(f);
            return *this;
        }
        client_base& operator=(shared_future<id_type> && f)
        {
            unregister_held_object();
            shared_state_ = hpx::traits::future_access<future_type>::
                get_shared_state(std::move(f));
            return *this;
        }
        client_base& operator=(future<id_type> && f)
        {
            unregister_held_object();
            shared_state_ = hpx::traits::future_access<future_type>::
                get_shared_state(std::move(f));
            return *this;
        }

        client_base& operator=(client_base const& rhs)
        {
            unregister_held_object();
            shared_state_ = rhs.shared_state_;
            return *this;
        }
        client_base& operator=(client_base && rhs)
        {
            unregister_held_object();
            shared_state_ = std::move(rhs.shared_state_);
            registered_name_ = std::move(rhs.registered_name_);
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

        ///////////////////////////////////////////////////////////////////////
        /// Create a new instance of an object on the locality as
        /// given by the parameter \a targetgid
        template <typename ...Ts>
        static Derived create(id_type const& targetgid, Ts&&... vs)
        {
            return Derived(stub_type::create_async(targetgid,
                std::forward<Ts>(vs)...));
        }

        template <typename ...Ts>
        static Derived create_colocated(id_type const& id, Ts&&... vs)
        {
            return Derived(stub_type::create_colocated_async(id,
                std::forward<Ts>(vs)...));
        }

        void free()
        {
            unregister_held_object();
            shared_state_.reset();
        }

        ///////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_COMPONENT_GET_GID_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        id_type const& get_gid() const
        {
            return get_id();
        }
#endif

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
            unregister_held_object();
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
            std::string const& symbolic_name)
        {
            hpx::agas::register_name(launch::sync, symbolic_name, f.get());
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

            HPX_ASSERT(registered_name_.empty());   // call only once
            if (manage_lifetime)
                registered_name_ = symbolic_name;

            typename hpx::traits::detail::shared_state_ptr<void>::type p =
                lcos::detail::make_continuation<void>(
                    *static_cast<Derived const*>(this), launch::all,
                    util::bind(&client_base::register_as_helper,
                        util::placeholders::_1, symbolic_name
                    ));
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
        // registered object anymore.
        void reset_registered_name()
        {
            HPX_ASSERT(!registered_name_.empty());   // call only once
            registered_name_.clear();
        }

        // Access registered name for the component
        std::string const& registered_name() const
        {
            return registered_name_;
        }

    protected:
        // will be set for created (non-attached) objects
        std::string registered_name_;

        // shared state holding the id_type this client refers to
        boost::intrusive_ptr<shared_state_type> shared_state_;
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

#endif

