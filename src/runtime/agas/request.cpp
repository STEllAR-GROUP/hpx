////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/serialize_sequence.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>

#include <boost/serialization/vector.hpp>

#include <boost/variant.hpp>
#include <boost/mpl/at.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/value_at.hpp>

namespace hpx { namespace agas
{
    struct request::request_data
    {
        request_data()
          : data(hpx::util::make_tuple())
        {}

        template <typename Tuple>
        request_data(const Tuple& tuple)
          : data(tuple)
        {}

        template <typename Tuple>
        request_data& operator=(const Tuple& tuple)
        {
            data = tuple;
            return *this;
        }

        int which() const
        {
            return data.which();
        }

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
          , subtype_parcel                  = 0xd
          // update HPX_AGAS_REQUEST_SUBTYPES above if you add more entries
        };

        // The types listed for any of the services represent the argument types
        // for this particular service.
        // The order of the variant types is significant, and should not be changed
        typedef boost::variant<
            // 0x0
            // primary_ns_change_credit
            util::tuple<
                naming::gid_type // lower
              , naming::gid_type // upper
              , boost::int64_t   // credit
            >
            // 0x1
            // primary_ns_unbind_gid
          , util::tuple<
                naming::gid_type // gid
              , boost::uint64_t  // count
            >
            // 0x2
            // primary_ns_bind_gid
          , util::tuple<
                naming::gid_type // gid
              , gva              // resolved address
            >
            // 0x3
            // primary_ns_resolve_gid
          , util::tuple<
                naming::gid_type // gid
            >
            // 0x4
            // primary_ns_allocate
          , util::tuple<
                naming::locality // locality
              , boost::uint64_t  // count
              , boost::uint32_t  // num_threads
              , naming::gid_type // suggested prefix
            >
            // 0x5
            // primary_ns_free
            // primary_ns_resolve_locality
          , util::tuple<
                naming::locality // locality
            >
            // 0x6
            // component_ns_resolve_id
            // component_ns_get_component_type
          , util::tuple<
                components::component_type // ctype
            >
            // 0x7
            // component_ns_bind_prefix
          , util::tuple<
                std::string     // name
              , boost::uint32_t // prefix
            >
            // 0x8
            // symbol_ns_bind
          , util::tuple<
                std::string      // name
              , naming::gid_type // gid
            >
            // 0x9
            // component_ns_bind_name
            // component_ns_unbind_name
            // symbol_ns_resolve
            // symbol_ns_unbind
            // component_ns_statistics
            // primary_ns_statistics
            // symbol_ns_statistics
          , util::tuple<
                std::string // name
            >
            // 0xa
            // symbol_ns_iterate_names
          , util::tuple<
                iterate_names_function_type // f
            >
            // 0xb
            // component_ns_iterate_types
          , util::tuple<
                iterate_types_function_type // f
            >
            // 0xc
            // primary_ns_localities
            // primary_ns_resolved_localities
          , util::tuple<
            >
            // 0xd
            // primary_ns_route
          , util::tuple<
                parcelset::parcel
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

        data_type data;
    };

    request::request()
        : mc(invalid_request)
        , data(new request_data(util::make_tuple()))
    {}

    request::request(
        namespace_action_code type_
      , naming::gid_type const& lower_
      , naming::gid_type const& upper_
      , boost::int64_t count_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(lower_, upper_, count_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , naming::gid_type const& gid_
      , boost::uint64_t count_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(gid_, count_)))
    {
        // TODO: verification of namespace_action_code
    }

    // REVIEW: Should the GVA here be a resolved address?
    request::request(
        namespace_action_code type_
      , naming::gid_type const& gid_
      , gva const& gva_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(gid_, gva_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , naming::gid_type const& gid_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(gid_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , naming::locality const& locality_
      , boost::uint64_t count_
      , boost::uint32_t num_threads_
      , naming::gid_type prefix_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(locality_, count_, num_threads_, prefix_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , naming::locality const& locality_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(locality_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , components::component_type ctype_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(ctype_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , std::string const& name_
      , boost::uint32_t prefix_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(name_, prefix_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , std::string const& name_
      , naming::gid_type const& gid_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(name_, gid_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , std::string const& name_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(name_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , iterate_names_function_type const& f_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(f_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , iterate_types_function_type const& f_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(f_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , parcelset::parcel const& p
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(p)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple()))
    {
        // TODO: verification of namespace_action_code
    }

    ///////////////////////////////////////////////////////////////////////////
    // copy constructor
    request::request(
        request const& other
        )
      : mc(other.mc)
      , data(new request_data(*other.data))
    {}

    // copy assignment
    request& request::operator=(
        request const& other
        )
    {
        mc = other.mc;
        data.reset(new request_data(*other.data));
        return *this;
    }

    gva request::get_gva(
        error_code& ec
        ) const
    {
        return data->get_data<request_data::subtype_gid_gva, 1>(ec);
    }

    boost::uint64_t request::get_count(
        error_code& ec
        ) const
    { // {{{
        switch (data->which())
        {
            case request_data::subtype_gid_count:
                return data->get_data<request_data::subtype_gid_count, 1>(ec);

            case request_data::subtype_locality_count:
                return data->get_data<request_data::subtype_locality_count, 1>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "request::get_count",
                    "invalid operation for request type");
                return 0;
            }
        };
    } // }}}

    boost::int64_t request::get_credit(
        error_code& ec
        ) const
    {
        return data->get_data<request_data::subtype_gid_gid_credit, 2>(ec);
    }

    components::component_type request::get_component_type(
        error_code& ec
        ) const
    {
        return data->get_data<request_data::subtype_ctype, 0>(ec);
    }

    boost::uint32_t request::get_locality_id(
        error_code& ec
        ) const
    {
        return data->get_data<request_data::subtype_name_prefix, 1>(ec);
    }

    request::iterate_names_function_type request::get_iterate_names_function(
        error_code& ec
        ) const
    {
        return data->get_data<request_data::subtype_iterate_names_function, 0>(ec);
    }

    request::iterate_types_function_type request::get_iterate_types_function(
        error_code& ec
        ) const
    {
        return data->get_data<request_data::subtype_iterate_types_function, 0>(ec);
    }

    parcelset::parcel request::get_parcel(
        error_code& ec
        ) const
    {
        return data->get_data<request_data::subtype_parcel, 0>(ec);
    }

    naming::locality request::get_locality(
        error_code& ec
        ) const
    { // {{{
        switch (data->which())
        {
            case request_data::subtype_locality_count:
                return data->get_data<request_data::subtype_locality_count, 0>(ec);

            case request_data::subtype_locality:
                return data->get_data<request_data::subtype_locality, 0>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "request::get_locality",
                    "invalid operation for request type");
                return naming::locality();
            }
        }
    } // }}}

    naming::gid_type request::get_gid(
        error_code& ec
        ) const
    { // {{{
        switch (data->which())
        {
            case request_data::subtype_gid:
                return data->get_data<request_data::subtype_gid, 0>(ec);

            case request_data::subtype_gid_gva:
                return data->get_data<request_data::subtype_gid_gva, 0>(ec);

            case request_data::subtype_gid_count:
                return data->get_data<request_data::subtype_gid_count, 0>(ec);

            case request_data::subtype_name_gid:
                return data->get_data<request_data::subtype_name_gid, 1>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "request::get_gid",
                    "invalid operation for request type");
                return naming::invalid_gid;
            }
        }
    } // }}}

    naming::gid_type request::get_lower_bound(
        error_code& ec
        ) const
    {
        return data->get_data<request_data::subtype_gid_gid_credit, 0>(ec);
    }

    naming::gid_type request::get_upper_bound(
        error_code& ec
        ) const
    {
        return data->get_data<request_data::subtype_gid_gid_credit, 1>(ec);
    }

    std::string request::get_name(
        error_code& ec
        ) const
    { // {{{
        switch (data->which())
        {
            case request_data::subtype_name:
                return data->get_data<request_data::subtype_name, 0>(ec);

            case request_data::subtype_name_prefix:
                return data->get_data<request_data::subtype_name_prefix, 0>(ec);

            case request_data::subtype_name_gid:
                return data->get_data<request_data::subtype_name_gid, 0>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "request::get_name",
                    "invalid operation for request type");
                return "";
            }
        };
    } // }}}

    std::string request::get_statistics_counter_name(
        error_code& ec
        ) const
    {
        return data->get_data<request_data::subtype_name, 0>(ec);
    }

    boost::uint32_t request::get_num_threads(
        error_code& ec
        ) const
    {
        return data->get_data<request_data::subtype_locality_count, 2>(ec);
    }

    naming::gid_type request::get_suggested_prefix(
        error_code& ec
        ) const
    {
        return data->get_data<request_data::subtype_locality_count, 3>(ec);
    }

    struct save_visitor : boost::static_visitor<void>
    {
      private:
        hpx::util::portable_binary_oarchive& ar;

      public:
        save_visitor(hpx::util::portable_binary_oarchive& ar_)
          : ar(ar_)
        {}

        template <
            typename Sequence
        >
        void operator()(Sequence const& seq) const
        {
            // TODO: verification?
            util::serialize_sequence(ar, seq);
        }
    };

    void request::save(hpx::util::portable_binary_oarchive& ar, const unsigned int) const
    { // {{{
        // TODO: versioning?
        int which = data->which();

        ar & which;
        ar & mc;
        boost::apply_visitor(save_visitor(ar), data->data);
    } // }}}

#define HPX_LOAD_SEQUENCE(z, n, _)                                            \
    case n:                                                                   \
        {                                                                     \
            typename boost::mpl::at_c<                                        \
                typename request_data::data_type::types, n                    \
            >::type d;                                                        \
            util::serialize_sequence(ar, d);                                  \
            data->data = d;                                                   \
            return;                                                           \
        }                                                                     \
    /**/

    void request::load(hpx::util::portable_binary_iarchive& ar, const unsigned int)
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
}}

HPX_UTIL_REGISTER_FUNCTION(
    void(std::string const&, hpx::naming::gid_type const&)
  , hpx::util::function<void(std::string const&, hpx::naming::gid_type const&)>
  , request_iterate_names_function_type
)

HPX_UTIL_REGISTER_FUNCTION(
    void(std::string const&, hpx::components::component_type)
  , hpx::util::function<void(std::string const&, hpx::components::component_type)>
  , request_iterate_types_function_type
)

