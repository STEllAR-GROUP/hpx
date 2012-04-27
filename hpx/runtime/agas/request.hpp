////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_AB01A9FE_45BE_43EF_B9AD_05B701B06685)
#define HPX_AB01A9FE_45BE_43EF_B9AD_05B701B06685

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/serialize_sequence.hpp>
#include <hpx/util/function.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/runtime/agas/gva.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/component_type.hpp>

#include <boost/variant.hpp>
#include <boost/mpl/at.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/fusion/include/at_c.hpp>

// The number of types that the request's variant can represent.
#define HPX_AGAS_REQUEST_SUBTYPES 13

namespace hpx { namespace agas
{

// TODO: Ensure that multiple invocations of get_data get optimized into the
// same jump table.
struct request
{
  public:
    typedef hpx::util::function<
        void(std::string const&, naming::gid_type const&)
    > iterate_names_function_type;

    typedef hpx::util::function<
        void(std::string const&, components::component_type)
    > iterate_types_function_type;

    request()
        : mc(invalid_request)
        , data(boost::fusion::make_vector())
    {}

    request(
        namespace_action_code type_
      , naming::gid_type const& lower_
      , naming::gid_type const& upper_
      , boost::int64_t count_
        )
      : mc(type_)
      , data(boost::fusion::make_vector(lower_, upper_, count_))
    {
        // TODO: verification of namespace_action_code
    }

    request(
        namespace_action_code type_
      , naming::gid_type const& gid_
      , boost::uint64_t count_
        )
      : mc(type_)
      , data(boost::fusion::make_vector(gid_, count_))
    {
        // TODO: verification of namespace_action_code
    }

    // REVIEW: Should the GVA here be a resolved address?
    request(
        namespace_action_code type_
      , naming::gid_type const& gid_
      , gva const& gva_
        )
      : mc(type_)
      , data(boost::fusion::make_vector(gid_, gva_))
    {
        // TODO: verification of namespace_action_code
    }

    request(
        namespace_action_code type_
      , naming::gid_type const& gid_
        )
      : mc(type_)
      , data(boost::fusion::make_vector(gid_))
    {
        // TODO: verification of namespace_action_code
    }

    request(
        namespace_action_code type_
      , naming::locality const& locality_
      , boost::uint64_t count_
        )
      : mc(type_)
      , data(boost::fusion::make_vector(locality_, count_))
    {
        // TODO: verification of namespace_action_code
    }

    request(
        namespace_action_code type_
      , naming::locality const& locality_
        )
      : mc(type_)
      , data(boost::fusion::make_vector(locality_))
    {
        // TODO: verification of namespace_action_code
    }

    request(
        namespace_action_code type_
      , components::component_type ctype_
        )
      : mc(type_)
      , data(boost::fusion::make_vector(ctype_))
    {
        // TODO: verification of namespace_action_code
    }

/*
    request(
        namespace_action_code type_
      , boost::int32_t ctype_
        )
      : mc(type_)
      , data(boost::fusion::make_vector(ctype_))
    {
        // TODO: verification of namespace_action_code
    }
*/

    request(
        namespace_action_code type_
      , std::string const& name_
      , boost::uint32_t prefix_
        )
      : mc(type_)
      , data(boost::fusion::make_vector(name_, prefix_))
    {
        // TODO: verification of namespace_action_code
    }

    request(
        namespace_action_code type_
      , std::string const& name_
      , naming::gid_type const& gid_
        )
      : mc(type_)
      , data(boost::fusion::make_vector(name_, gid_))
    {
        // TODO: verification of namespace_action_code
    }

    request(
        namespace_action_code type_
      , std::string const& name_
        )
      : mc(type_)
      , data(boost::fusion::make_vector(name_))
    {
        // TODO: verification of namespace_action_code
    }

    request(
        namespace_action_code type_
      , iterate_names_function_type const& f_
        )
      : mc(type_)
      , data(boost::fusion::make_vector(f_))
    {
        // TODO: verification of namespace_action_code
    }

    request(
        namespace_action_code type_
      , iterate_types_function_type const& f_
        )
      : mc(type_)
      , data(boost::fusion::make_vector(f_))
    {
        // TODO: verification of namespace_action_code
    }

    explicit request(
        namespace_action_code type_
        )
      : mc(type_)
      , data(boost::fusion::make_vector())
    {
        // TODO: verification of namespace_action_code
    }

    // copy constructor
    request(
        request const& other
        )
      : mc(other.mc)
      , data(other.data)
    {}

    // copy assignment
    request& operator=(
        request const& other
        )
    {
        mc = other.mc;
        data = other.data;
        return *this;
    }

    gva get_gva(
        error_code& ec = throws
        ) const
    {
        return get_data<subtype_gid_gva, 1>(ec);
    }

    boost::uint64_t get_count(
        error_code& ec = throws
        ) const
    { // {{{
        switch (data.which())
        {
            case subtype_gid_count:
                return get_data<subtype_gid_count, 1>(ec);

            case subtype_locality_count:
                return get_data<subtype_locality_count, 1>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "request::get_count",
                    "invalid operation for request type");
                return 0;
            }
        };
    } // }}}

    boost::int64_t get_credit(
        error_code& ec = throws
        ) const
    {
        return get_data<subtype_gid_gid_credit, 2>(ec);
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
        return get_data<subtype_name_prefix, 1>(ec);
    }

    iterate_names_function_type get_iterate_names_function(
        error_code& ec = throws
        ) const
    {
        return get_data<subtype_iterate_names_function, 0>(ec);
    }

    iterate_types_function_type get_iterate_types_function(
        error_code& ec = throws
        ) const
    {
        return get_data<subtype_iterate_types_function, 0>(ec);
    }

    naming::locality get_locality(
        error_code& ec = throws
        ) const
    { // {{{
        naming::locality l;

        // Don't let the first attempt throw.
        error_code first_try;
        l = get_data<subtype_locality_count, 0>(first_try);

        // If the first try failed, check again.
        if (first_try)
            l = get_data<subtype_locality, 0>(ec);
        else if (&ec != &throws)
            ec = make_success_code();

        return l;
    } // }}}

    naming::gid_type get_gid(
        error_code& ec = throws
        ) const
    { // {{{
        switch (data.which())
        {
            case subtype_gid:
                return get_data<subtype_gid, 0>(ec);

            case subtype_gid_gva:
                return get_data<subtype_gid_gva, 0>(ec);

            case subtype_gid_count:
                return get_data<subtype_gid_count, 0>(ec);

            case subtype_name_gid:
                return get_data<subtype_name_gid, 1>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "request::get_gid",
                    "invalid operation for request type");
                return naming::invalid_gid;
            }
        };
    } // }}}

    naming::gid_type get_lower_bound(
        error_code& ec = throws
        ) const
    {
        return get_data<subtype_gid_gid_credit, 0>(ec);
    }

    naming::gid_type get_upper_bound(
        error_code& ec = throws
        ) const
    {
        return get_data<subtype_gid_gid_credit, 1>(ec);
    }

    std::string get_name(
        error_code& ec = throws
        ) const
    { // {{{
        switch (data.which())
        {
            case subtype_name:
                return get_data<subtype_name, 0>(ec);

            case subtype_name_prefix:
                return get_data<subtype_name_prefix, 0>(ec);

            case subtype_name_gid:
                return get_data<subtype_name_gid, 0>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "request::get_name",
                    "invalid operation for request type");
                return "";
            }
        };
    } // }}}

    namespace_action_code get_action_code() const
    {
        return mc;
    }

  private:
    friend class boost::serialization::access;

    enum subtype
    {
        subtype_gid_gid_credit          = 0x0
      , subtype_gid_count               = 0x1
      , subtype_gid_gva                 = 0x2
      , subtype_gid                     = 0x3
      , subtype_locality_count          = 0x4
      , subtype_locality                = 0x5
      , subtype_ctype                   = 0x6
      , subtype_name_prefix             = 0x7
      , subtype_name_gid                = 0x8
      , subtype_name                    = 0x9
      , subtype_iterate_names_function  = 0xa
      , subtype_iterate_types_function  = 0xb
      , subtype_void                    = 0xc
    };

    // The order of the variant types is significant, and should not be changed
    typedef boost::variant<
        // 0x0
        // primary_ns_change_credit
        boost::fusion::vector3<
            naming::gid_type // lower
          , naming::gid_type // upper
          , boost::int64_t   // credit
        >
        // 0x1
        // primary_ns_unbind_gid
      , boost::fusion::vector2<
            naming::gid_type // gid
          , boost::uint64_t  // count
        >
        // 0x2
        // primary_ns_bind_gid
      , boost::fusion::vector2<
            naming::gid_type // gid
          , gva              // resolved address
        >
        // 0x3
        // primary_ns_resolve_gid
      , boost::fusion::vector1<
            naming::gid_type // gid
        >
        // 0x4
        // primary_ns_allocate
      , boost::fusion::vector2<
            naming::locality // locality
          , boost::uint64_t  // count
        >
        // 0x5
        // primary_ns_free
        // primary_ns_resolve_locality
      , boost::fusion::vector1<
            naming::locality // locality
        >
        // 0x6
        // component_ns_resolve_id
      , boost::fusion::vector1<
            components::component_type // ctype
        >
        // 0x7
        // component_ns_bind_prefix
      , boost::fusion::vector2<
            std::string     // name
          , boost::uint32_t // prefix
        >
        // 0x8
        // symbol_ns_bind
      , boost::fusion::vector2<
            std::string      // name
          , naming::gid_type // gid
        >
        // 0x9
        // component_ns_bind_name
        // component_ns_unbind
        // symbol_ns_resolve
        // symbol_ns_unbind
      , boost::fusion::vector1<
            std::string // name
        >
        // 0xa
        // symbol_ns_iterate_names
      , boost::fusion::vector1<
            iterate_names_function_type // f
        >
        // 0xb
        // component_ns_iterate_types
      , boost::fusion::vector1<
            iterate_types_function_type // f
        >
        // 0xc
        // primary_ns_localities
      , boost::fusion::vector0<
        >
    > data_type;

    // {{{ variant helper TODO: consolidate with helpers in response
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
                      , "request::get_data"
                      , "internal data corruption");
                    return return_type();
                }

                if (&ec != &throws)
                    ec = make_success_code();

                return boost::fusion::at_c<N>(*v);
            }

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "request::get_data",
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

        // Build the jump table.
        switch (which)
        {
            BOOST_PP_REPEAT(HPX_AGAS_REQUEST_SUBTYPES, HPX_LOAD_SEQUENCE, _)

            default: {
                HPX_THROW_EXCEPTION(invalid_data,
                    "request::load",
                    "unknown or invalid data loaded");
                return;
            }
        };
    } // }}}

#undef HPX_LOAD_SEQUENCE

    BOOST_SERIALIZATION_SPLIT_MEMBER()

    namespace_action_code mc;
    data_type data;
};

}}

#ifdef __GNUG__
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
BOOST_CLASS_VERSION(hpx::agas::request, HPX_AGAS_VERSION)
BOOST_CLASS_TRACKING(hpx::agas::request, boost::serialization::track_never)
#ifdef __GNUG__
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif

#endif // HPX_AB01A9FE_45BE_43EF_B9AD_05B701B06685

