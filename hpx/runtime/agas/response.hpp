////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_FB40C7A4_33B0_4C64_A16B_2A3FEEB237ED)
#define HPX_FB40C7A4_33B0_4C64_A16B_2A3FEEB237ED

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/serialize_sequence.hpp>
#include <hpx/traits/get_remote_result.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/runtime/agas/gva.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>

#include <boost/variant.hpp>
#include <boost/mpl/at.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/tracking.hpp>

// The number of types that response's variant can represent.
#define HPX_AGAS_RESPONSE_SUBTYPES 9

namespace hpx { namespace agas
{

// TODO: Ensure that multiple invocations of get_data get optimized into the
// same jump table.
struct response
{
  public:
    response()
        : mc(invalid_request)
        , status(invalid_status)
        , data(boost::fusion::make_vector())
    {}

    response(
        namespace_action_code type_
      , naming::gid_type lower_
      , naming::gid_type upper_
      , boost::uint32_t prefix_
      , error status_ = success
        )
      : mc(type_)
      , status(status_)
      , data(boost::fusion::make_vector(lower_, upper_, prefix_))
    {
        // TODO: verification of namespace_action_code
    }

    response(
        namespace_action_code type_
      , naming::gid_type const& gidbase_
      , gva const& gva_
      , error status_ = success
        )
      : mc(type_)
      , status(status_)
      , data(boost::fusion::make_vector(gidbase_, gva_))
    {
        // TODO: verification of namespace_action_code
    }

    response(
        namespace_action_code type_
      , gva const& gva_
      , error status_ = success
        )
      : mc(type_)
      , status(status_)
      , data(boost::fusion::make_vector(gva_))
    {
        // TODO: verification of namespace_action_code
    }

    response(
        namespace_action_code type_
      , components::component_type ctype_
      , error status_ = success
        )
      : mc(type_)
      , status(status_)
      , data(boost::fusion::make_vector(ctype_))
    {
        // TODO: verification of namespace_action_code
    }

/*
    response(
        namespace_action_code type_
      , components::component_type ctype_
      , error status_ = success
        )
      : mc(type_)
      , status(status_)
      , data(boost::fusion::make_vector(ctype_))
    {
        // TODO: verification of namespace_action_code
    }
*/

    response(
        namespace_action_code type_
      , std::vector<boost::uint32_t> const& prefixes_
      , error status_ = success
        )
      : mc(type_)
      , status(status_)
      , data(boost::fusion::make_vector(prefixes_))
    {
        // TODO: verification of namespace_action_code
    }

    response(
        namespace_action_code type_
      , naming::gid_type gid_
      , error status_ = success
        )
      : mc(type_)
      , status(status_)
      , data(boost::fusion::make_vector(gid_))
    {
        // TODO: verification of namespace_action_code
    }

    response(
        namespace_action_code type_
      , boost::uint32_t prefix_
      , error status_ = success
        )
      : mc(type_)
      , status(status_)
      , data(boost::fusion::make_vector(prefix_))
    {
        // TODO: verification of namespace_action_code
    }

    response(
        namespace_action_code type_
      , std::string const& name_
      , error status_ = success
        )
      : mc(type_)
      , status(status_)
      , data(boost::fusion::make_vector(name_))
    {
        // TODO: verification of namespace_action_code
    }

    explicit response(
        namespace_action_code type_
      , error status_ = success
        )
      : mc(type_)
      , status(status_)
      , data(boost::fusion::make_vector())
    {
        // TODO: verification of namespace_action_code
    }

    ///////////////////////////////////////////////////////////////////////////
    // copy constructor
    response(
        response const& other
        )
      : mc(other.mc)
      , status(other.status)
      , data(other.data)
    {}

    // copy assignment
    response& operator=(
        response const& other
        )
    {
        if (this != &other)
        {
            mc = other.mc;
            status = other.status;
            data = other.data;
        }
        return *this;
    }

    gva get_gva(
        error_code& ec = throws
        ) const
    {
        gva g;

        // Don't let the first attempt throw.
        error_code first_try;
        g = get_data<subtype_gid_gva, 1>(first_try);

        // If the first try failed, check again.
        if (first_try)
            g = get_data<subtype_gva, 0>(ec);
        else if (&ec != &throws)
            ec = make_success_code();

        return g;
    }

    std::vector<boost::uint32_t> get_localities(
        error_code& ec = throws
        ) const
    {
        return get_data<subtype_prefixes, 0>(ec);
    }

    components::component_type get_component_type(
        error_code& ec = throws
        ) const
    {
        return get_data<subtype_ctype, 0>(ec);
    }

    boost::uint32_t get_locality_id(
        error_code& ec = throws
        ) const
    {
        boost::uint32_t prefix;

        // Don't let the first attempt throw.
        error_code first_try;
        prefix = get_data<subtype_gid_gid_prefix, 2>(first_try);

        // If the first try failed, check again.
        if (first_try)
            prefix = get_data<subtype_prefix, 0>(ec);
        else if (&ec != &throws)
            ec = make_success_code();

        return prefix;
    }

    naming::gid_type get_base_gid(
        error_code& ec = throws
        ) const
    {
        return get_data<subtype_gid_gva, 0>(ec);
    }

    naming::gid_type get_gid(
        error_code& ec = throws
        ) const
    {
        return get_data<subtype_gid, 0>(ec);
    }

    naming::gid_type get_lower_bound(
        error_code& ec = throws
        ) const
    {
        return get_data<subtype_gid_gid_prefix, 0>(ec);
    }

    naming::gid_type get_upper_bound(
        error_code& ec = throws
        ) const
    {
        return get_data<subtype_gid_gid_prefix, 1>(ec);
    }

    std::string get_component_typename(
        error_code& ec = throws
        ) const
    {
        return get_data<subtype_string, 0>(ec);
    }

    naming::gid_type get_statistics_counter(
        error_code& ec = throws
        ) const
    {
        return get_data<subtype_gid, 0>(ec);
    }

    namespace_action_code get_action_code() const
    {
        return mc;
    }

    error get_status() const
    {
        return status;
    }

  private:
    friend class boost::serialization::access;

    enum subtype
    {
        subtype_gid_gid_prefix  = 0x0
      , subtype_gid_gva         = 0x1
      , subtype_gva             = 0x2
      , subtype_ctype           = 0x3
      , subtype_prefixes        = 0x4
      , subtype_gid             = 0x5
      , subtype_prefix          = 0x6
      , subtype_void            = 0x7
      , subtype_string          = 0x8
      // update HPX_AGAS_RESPONSE_SUBTYPES is you add more subtypes
    };

    // The order of the variant types is significant, and should not be changed
    typedef boost::variant<
        // 0x0
        // primary_ns_allocate
        boost::fusion::vector3<
            naming::gid_type // lower bound
          , naming::gid_type // upper bound
          , boost::uint32_t  // prefix
        >
        // 0x1
        // primary_ns_resolve_gid
      , boost::fusion::vector2<
            naming::gid_type // idbase
          , gva              // gva
        >
        // 0x2
        // primary_ns_unbind_gid
      , boost::fusion::vector1<
            gva // gva
        >
        // 0x3
        // component_ns_bind_prefix
        // component_ns_bind_name
      , boost::fusion::vector1<
            components::component_type // ctype
        >
        // 0x4
        // primary_ns_localities
        // component_ns_resolve_id
      , boost::fusion::vector1<
            std::vector<boost::uint32_t> // prefixes
        >
        // 0x5
        // symbol_ns_unbind
        // symbol_ns_resolve
        // primary_ns_statistics
        // component_ns_statistics
        // symbol_ns_statistics
      , boost::fusion::vector1<
            naming::gid_type // gid
        >
        // 0x6
        // primary_ns_resolve_locality
      , boost::fusion::vector1<
            boost::uint32_t // prefix
        >
        // 0x7
        // primary_ns_free
        // primary_ns_bind_gid
        // component_ns_unbind_name
        // component_ns_iterate_types
        // symbol_ns_bind
        // symbol_ns_iterate_names
        // primary_ns_change_credit
      , boost::fusion::vector0<
        >
        // 0x8
        // component_ns_get_component_typename
      , boost::fusion::vector1<
            std::string   // component typename
        >
    > data_type;

    // {{{ variant helper TODO: consolidate with helpers in request
    template <
        subtype Type
      , int N
    >
    typename boost::fusion::result_of::value_at_c<
        typename boost::mpl::at_c<
            typename data_type::types, Type
        >::type, N
    >::type
    get_data(
        error_code& ec = throws
        ) const
    { // {{{
        typedef typename boost::mpl::at_c<
            typename data_type::types, Type
        >::type vector_type;

        typedef typename boost::fusion::result_of::value_at_c<
            vector_type, N
        >::type return_type;

        switch (data.which())
        {
            case Type:
            {
                vector_type const* v = boost::get<vector_type>(&data);

                if (!v)
                {
                    HPX_THROWS_IF(ec, invalid_data
                      , "response::get_data"
                      , "internal data corruption");
                    return return_type();
                }

                if (&ec != &throws)
                    ec = make_success_code();

                return boost::fusion::at_c<N>(*v);
            }

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "response::get_data",
                    "invalid operation for request type");
                return return_type();
            }
        };
    } // }}}
    // }}}

    template <
        typename Archive
    >
    struct save_visitor : boost::static_visitor<void>
    {
      private:
        Archive& ar;

      public:
        save_visitor(
            Archive& ar_
            )
          : ar(ar_)
        {}

        template <
            typename Sequence
        >
        void operator()(
            Sequence const& seq
            ) const
        {
            // TODO: verification?
            util::serialize_sequence(ar, seq);
        }
    };

    template <
        typename Archive
    >
    void save(
        Archive& ar
      , const unsigned int
        ) const
    { // {{{
        // TODO: versioning?
        int which = data.which();

        ar & which;
        ar & mc;
        ar & status;
        boost::apply_visitor(save_visitor<Archive>(ar), data);
    } // }}}

#define HPX_LOAD_SEQUENCE(z, n, _)                                          \
    case n:                                                                 \
        {                                                                   \
            typename boost::mpl::at_c<                                      \
                typename data_type::types, n                                \
            >::type d;                                                      \
            util::serialize_sequence(ar, d);                                \
            data = d;                                                       \
            return;                                                         \
        }                                                                   \
    /**/

    template <
        typename Archive
    >
    void load(
        Archive& ar
      , const unsigned int
        )
    { // {{{
        // TODO: versioning
        int which = -1;

        ar & which;
        ar & mc;
        ar & status;

        // Build the jump table.
        switch (which)
        {
            BOOST_PP_REPEAT(HPX_AGAS_RESPONSE_SUBTYPES, HPX_LOAD_SEQUENCE, _)

            default: {
                HPX_THROW_EXCEPTION(invalid_data,
                    "response::load",
                    "unknown or invalid data loaded");
                return;
            }
        };
    } // }}}

#undef HPX_LOAD_SEQUENCE

    BOOST_SERIALIZATION_SPLIT_MEMBER()

    namespace_action_code mc;
    error status;
    data_type data;
};

}

namespace traits
{

// TODO: verification of namespace_action_code
template <>
struct get_remote_result<naming::id_type, agas::response>
{
    static naming::id_type call(
        agas::response const& rep
        )
    {
        naming::gid_type raw_gid = rep.get_gid();

        if (naming::get_credit_from_gid(raw_gid) != 0)
            return naming::id_type(raw_gid, naming::id_type::managed);
        else
            return naming::id_type(raw_gid, naming::id_type::unmanaged);
    }
};

// TODO: verification of namespace_action_code
template <>
struct get_remote_result<bool, agas::response>
{
    static bool call(
        agas::response const& rep
        )
    {
        return success != rep.get_status();
    }
};

}}

#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
BOOST_CLASS_VERSION(hpx::agas::response, HPX_AGAS_VERSION)
BOOST_CLASS_TRACKING(hpx::agas::response, boost::serialization::track_never)
#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<hpx::agas::response>::set_value_action,
    set_value_action_agas_response_type)

namespace hpx { namespace agas { namespace create_result_ns {
    typedef
        hpx::lcos::base_lco_with_value<bool, hpx::agas::response>
        base_lco_bool_response_type;
    typedef
        hpx::lcos::base_lco_with_value<hpx::naming::id_type, hpx::agas::response>
        base_lco_id_type_response_type;
    typedef
        hpx::lcos::base_lco_with_value<std::vector<hpx::agas::response> >
        base_lco_vector_response_type;
}}}
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::agas::create_result_ns::base_lco_bool_response_type::set_value_action,
    set_value_action_agas_bool_response_type)

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::agas::create_result_ns::base_lco_id_type_response_type::set_value_action,
    set_value_action_agas_id_type_response_type)

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::agas::create_result_ns::base_lco_vector_response_type::set_value_action,
    set_value_action_agas_vector_response_type)

#endif // HPX_FB40C7A4_33B0_4C64_A16B_2A3FEEB237ED

