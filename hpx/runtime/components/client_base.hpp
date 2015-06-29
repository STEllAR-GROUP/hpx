//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CLIENT_BASE_OCT_31_2008_0424PM)
#define HPX_COMPONENTS_CLIENT_BASE_OCT_31_2008_0424PM

#include <hpx/config.hpp>
#include <hpx/traits/is_client.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/safe_bool.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/runtime/agas/interface.hpp>

#include <utility>
#include <vector>

#include <boost/utility/enable_if.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/has_xxx.hpp>
#include <boost/type_traits/is_base_of.hpp>

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
        typename util::always_void<typename Derived::stub_argument_type>::type
    > : boost::is_base_of<
            components::client_base<
                Derived, typename Derived::stub_argument_type
            >,
            Derived>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived>
    struct is_future<Derived,
        typename boost::enable_if_c<is_client<Derived>::value>::type>
      : boost::mpl::true_
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived>
    struct future_traits<Derived,
        typename boost::enable_if_c<is_client<Derived>::value>::type>
    {
        typedef naming::id_type type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived>
    struct future_access<Derived,
        typename boost::enable_if_c<is_client<Derived>::value>::type>
    {
        HPX_FORCEINLINE static
        typename traits::detail::shared_state_ptr<naming::id_type>::type const&
        get_shared_state(Derived const& client)
        {
            return client.share().shared_state_;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived>
    struct acquire_future_impl<Derived,
        typename boost::enable_if_c<is_client<Derived>::value>::type>
    {
        typedef Derived type;

        template <typename T_>
        HPX_FORCEINLINE
        Derived operator()(T_ && value) const
        {
            return std::forward<T_>(value);
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    namespace detail
    {
        BOOST_MPL_HAS_XXX_TRAIT_DEF(server_component_type)

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

        ///////////////////////////////////////////////////////////////////////
        template <typename Derived>
        struct client_unwrapper
        {
            typedef shared_future<naming::id_type> result_type;

            HPX_FORCEINLINE result_type
            operator()(future<Derived> f) const
            {
                return f.get().share();
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename Stub>
    class client_base : public detail::make_stub<Stub>::type
    {
    protected:
        typedef typename detail::make_stub<Stub>::type stub_type;
        typedef shared_future<naming::id_type> future_type;

    public:
        typedef Stub stub_argument_type;
        typedef typename detail::make_stub<Stub>::server_component_type
            server_component_type;

        client_base()
          : gid_()
        {}

        explicit client_base(naming::id_type const& gid)
          : gid_(lcos::make_ready_future(gid))
        {}
        explicit client_base(naming::id_type && gid)
          : gid_(lcos::make_ready_future(std::move(gid)))
        {}

        explicit client_base(future_type const& gid)
          : gid_(gid)
        {}
        explicit client_base(future_type && gid)
          : gid_(std::move(gid))
        {}
        explicit client_base(future<naming::id_type> && gid)
          : gid_(gid.share())
        {}

        client_base(client_base const& rhs)
          : gid_(rhs.gid_)
        {}
        client_base(client_base && rhs)
          : registered_name_(std::move(rhs.registered_name_)),
            gid_(std::move(rhs.gid_))
        {}

        // A future to a client_base can be unwrapped to represent the
        // client_base directly as a client_base holds another future to the
        // id of the referenced object.
        client_base(future<Derived> && d)
          : gid_(d.then(detail::client_unwrapper<Derived>()))
        {}

        ~client_base()
        {
            if (!registered_name_.empty())
            {
                error_code ec;      // ignore all exceptions
                agas::unregister_name_sync(registered_name_, ec);
            }
        }

        // copy assignment and move assignment
        client_base& operator=(naming::id_type const & gid)
        {
            gid_ = lcos::make_ready_future(gid);
            return *this;
        }
        client_base& operator=(naming::id_type && gid)
        {
            gid_ = lcos::make_ready_future(std::move(gid));
            return *this;
        }

        client_base& operator=(future_type const & gid)
        {
            gid_ = gid;
            return *this;
        }
        client_base& operator=(future_type && gid)
        {
            gid_ = std::move(gid);
            return *this;
        }
        client_base& operator=(future<naming::id_type> && gid)
        {
            gid_ = gid.share();
            return *this;
        }

        client_base& operator=(client_base const & rhs)
        {
            gid_ = rhs.gid_;
            registered_name_.clear();
            return *this;
        }
        client_base& operator=(client_base && rhs)
        {
            gid_ = std::move(rhs.gid_);
            registered_name_ = std::move(rhs.registered_name_);
            return *this;
        }

        bool valid() const
        {
            return gid_.valid() && !gid_.has_exception();
        }

        // check whether the embedded future is valid
        operator typename util::safe_bool<client_base>::result_type() const
        {
            return util::safe_bool<client_base>()(valid());
        }

        ///////////////////////////////////////////////////////////////////////
        /// Create a new instance of an object on the locality as
        /// given by the parameter \a targetgid
        template <typename ...Ts>
        static Derived create(naming::id_type const& targetgid, Ts&&... vs)
        {
            return Derived(stub_type::create_async(targetgid,
                std::forward<Ts>(vs)...));
        }

        template <typename ...Ts>
        static Derived create_colocated(naming::id_type const& id, Ts&&... vs)
        {
            return Derived(stub_type::create_colocated_async(id,
                std::forward<Ts>(vs)...));
        }

        void free()
        {
            gid_ = future_type();
        }

        ///////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_COMPONENT_GET_GID_COMPATIBILITY)
        naming::id_type const & get_gid() const
        {
            return gid_.get();
        }

#endif
        naming::id_type const & get_id() const
        {
            return gid_.get();
        }

        naming::gid_type const & get_raw_gid() const
        {
            return gid_.get().get_gid();
        }

        ///////////////////////////////////////////////////////////////////////
        future_type detach()
        {
            shared_future<naming::id_type> g;
            std::swap(gid_, g);
            return g;
        }

        future_type const& share() const
        {
            return gid_;
        }

        void reset(id_type const& id)
        {
            gid_ = make_ready_future(id);
        }

        void reset(id_type && id)
        {
            gid_ = make_ready_future(std::move(id));
        }

        void reset(future_type && rhs)
        {
            gid_ = std::move(rhs);
        }

        ///////////////////////////////////////////////////////////////////////
        // Interface mimicking future
        bool is_ready() const
        {
            return gid_.is_ready();
        }

        void wait() const
        {
            return gid_.wait();
        }

        ///////////////////////////////////////////////////////////////////////
    protected:
        static void register_as_helper(shared_future<naming::id_type> && f,
            std::string const& symbolic_name)
        {
            hpx::agas::register_name(symbolic_name, f.get());
        }

        void reset_registered_name()
        {
            registered_name_.clear();
        }

    public:
        // Register our id with AGAS using the given name
        future<void> register_as(std::string const& symbolic_name)
        {
            HPX_ASSERT(registered_name_.empty());   // call only once
            registered_name_ = symbolic_name;

            return gid_.then(util::bind(&client_base::register_as_helper,
                util::placeholders::_1, symbolic_name));
        }

        // Retrieve the id associated with the given name and use it to
        // initialize this client_base instance.
        void connect_to(std::string const& symbolic_name)
        {
            gid_ = agas::on_symbol_namespace_event(symbolic_name,
                agas::symbol_ns_bind, true).share();
        }

    protected:
        template <typename F>
        static typename lcos::detail::future_then_result<Derived, F>::cont_result
        on_ready(future_type && fut, F f)
        {
            return f(Derived(std::move(fut)));
        }

    public:
        template <typename F>
        typename lcos::detail::future_then_result<
            Derived, typename util::decay<F>::type
        >::type
        then(F && f)
        {
            typedef typename util::decay<F>::type func_type;
            return gid_.then(util::bind(
                util::one_shot(&client_base::template on_ready<func_type>),
                util::placeholders::_1, std::forward<F>(f)));
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & gid_;
        }

    protected:
        // will be set for created (non-attached) objects
        std::string registered_name_;
        future_type gid_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Client>
    inline typename boost::enable_if_c<
        traits::is_client<Client>::value, Client
    >::type
    make_client(hpx::id_type const& id)
    {
        return Client(id);
    }

    template <typename Client>
    inline typename boost::enable_if_c<
        traits::is_client<Client>::value, Client
    >::type
    make_client(hpx::future<hpx::id_type> const& id)
    {
        return Client(id);
    }

    template <typename Client>
    inline typename boost::enable_if_c<
        traits::is_client<Client>::value, Client
    >::type
    make_client(hpx::future<hpx::id_type> && id)
    {
        return Client(std::move(id));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Client>
    inline typename boost::enable_if_c<
        traits::is_client<Client>::value, std::vector<Client>
    >::type
    make_client(std::vector<hpx::id_type> const& ids)
    {
        std::vector<Client> result;
        result.reserve(ids.size());
        for (hpx::id_type const& id: ids)
        {
            result.push_back(Client(id));
        }
        return result;
    }

    template <typename Client>
    inline typename boost::enable_if_c<
        traits::is_client<Client>::value, std::vector<Client>
    >::type
    make_client(std::vector<hpx::future<hpx::id_type> > const& ids)
    {
        std::vector<Client> result;
        result.reserve(ids.size());
        for (hpx::future<hpx::id_type> const& id: ids)
        {
            result.push_back(Client(id));
        }
        return result;
    }

    template <typename Client>
    inline typename boost::enable_if_c<
        traits::is_client<Client>::value, std::vector<Client>
    >::type
    make_client(std::vector<hpx::future<hpx::id_type> > && ids)
    {
        std::vector<Client> result;
        result.reserve(ids.size());
        for (hpx::future<hpx::id_type>& id: ids)
        {
            result.push_back(Client(std::move(id)));
        }
        return result;
    }
}}

#endif

