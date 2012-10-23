//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CLIENT_BASE_OCT_31_2008_0424PM)
#define HPX_COMPONENTS_CLIENT_BASE_OCT_31_2008_0424PM

#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/util/move.hpp>
#include <hpx/lcos/future.hpp>

#include <boost/move/move.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    namespace detail
    {
        BOOST_MPL_HAS_XXX_TRAIT_DEF(server_component_type)

        // wrap a give type such that it is usable as a stub_base
        template <typename Stub, typename Enable = void>
        struct make_stub
        {
            typedef components::stub_base<Stub> type;
        };

        template <typename Stub>
        struct make_stub<Stub, 
            typename boost::enable_if<has_server_component_type<Stub> >::type>
        {
            typedef Stub type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename Stub>
    class client_base : public detail::make_stub<Stub>::type
    {
    private:
        BOOST_COPYABLE_AND_MOVABLE(client_base);

    protected:
        typedef typename detail::make_stub<Stub>::type stub_type;
        typedef future<naming::id_type> future_type;

    public:
        client_base()
          : gid_()
        {}

        explicit client_base(naming::id_type const& gid)
          : gid_(future_type(gid))
        {}
        explicit client_base(BOOST_RV_REF(naming::id_type) gid)
          : gid_(future_type(boost::move(gid)))
        {}

        explicit client_base(future_type const& gid)
          : gid_(gid)
        {}
        explicit client_base(BOOST_RV_REF(future_type) gid)
          : gid_(boost::move(gid))
        {}

        explicit client_base(client_base const& rhs)
          : gid_(rhs.gid_)
        {}
        explicit client_base(BOOST_RV_REF(client_base) rhs)
          : gid_(boost::move(rhs.gid_))
        {}

        // copy assignment and move assignment
        client_base& operator=(BOOST_COPY_ASSIGN_REF(naming::id_type) gid)
        {
            gid_ = lcos::make_future(gid);
            return *this;
        }
        client_base& operator=(BOOST_RV_REF(naming::id_type) gid)
        {
            gid_ = lcos::make_future(boost::move(gid));
            return *this;
        }

        client_base& operator=(BOOST_COPY_ASSIGN_REF(future_type) gid)
        {
            gid_ = gid;
            return *this;
        }
        client_base& operator=(BOOST_RV_REF(future_type) gid)
        {
            gid_ = boost::move(gid);
            return *this;
        }

        client_base& operator=(BOOST_COPY_ASSIGN_REF(client_base) rhs)
        {
            if (this != &rhs)
                gid_ = rhs.gid_;
            return *this;
        }
        client_base& operator=(BOOST_RV_REF(client_base) rhs)
        {
            if (this != &rhs)
                gid_ = boost::move(rhs.gid_);
            return *this;
        }

        ///////////////////////////////////////////////////////////////////////
        /// Create a new instance of an object on the locality as
        /// given by the parameter \a targetgid
        Derived& create(naming::id_type const& targetgid)
        {
            gid_ = stub_type::create_async(targetgid);
            return static_cast<Derived&>(*this);
        }

#define HPX_CLIENT_BASE_CREATE(Z, N, D)                                       \
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>                      \
        Derived& create(naming::id_type const& targetgid,                     \
            HPX_ENUM_FWD_ARGS(N, Arg, arg))                                   \
        {                                                                     \
            gid_ = stub_type::create_async(targetgid,                         \
                HPX_ENUM_FORWARD_ARGS(N , Arg, arg));                         \
            return static_cast<Derived&>(*this);                              \
        }                                                                     \
    /**/

        BOOST_PP_REPEAT_FROM_TO(
            1
          , HPX_ACTION_ARGUMENT_LIMIT
          , HPX_CLIENT_BASE_CREATE
          , _
        )

#undef HPX_CLIENT_BASE_CREATE

        void free()
        {
        }

        ///////////////////////////////////////////////////////////////////////
        naming::id_type get_gid() const
        {
            return gid_.get();
        }

        naming::gid_type get_raw_gid() const
        {
            return gid_.get().get_gid();
        }

        future_type detach()
        {
            future<naming::id_type> g;
            std::swap(gid_, g);
            return g;
        }

    private:
        friend class boost::serialization::access;
        
        template <typename Archive>
        void load(Archive & ar, unsigned)
        {
            naming::id_type id;
            ar & id;

            gid_ = lcos::make_future(id);
        }

        template <typename Archive>
        void save(Archive & ar, unsigned) const
        {
            naming::id_type id = gid_.get();
            ar & id;
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()

        future_type gid_;
    };
}}

#endif

