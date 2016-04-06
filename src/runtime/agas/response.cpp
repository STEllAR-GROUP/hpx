////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_fwd.hpp>

#include <vector>

#include <hpx/hpx.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/base_lco_factory.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/util/tuple.hpp>

// The number of types that response's variant can represent.
#define HPX_AGAS_RESPONSE_SUBTYPES 12

namespace hpx { namespace agas
{
    struct response::response_data
    {
        response_data()
          : data(hpx::util::make_tuple())
        {}

        template <typename Tuple>
        response_data(Tuple const & tuple)
          : data(tuple)
        {}

        template <typename Tuple>
        response_data& operator=(Tuple const & tuple)
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
            subtype_gid_gid_prefix      = 0x0
          , subtype_gid_gva_prefix      = 0x1
          , subtype_gva                 = 0x2
          , subtype_ctype               = 0x3
          , subtype_prefixes            = 0x4
          , subtype_num_threads         = 0x4
          , subtype_gid                 = 0x5
          , subtype_statistics_counter  = 0x5
          , subtype_prefix              = 0x6
          , subtype_void                = 0x7
          , subtype_string              = 0x8
          , subtype_resolved_localities = 0x9
          , subtype_added_credits       = 0xa
          , subtype_endpoints           = 0xb
          // update HPX_AGAS_RESPONSE_SUBTYPES is you add more subtypes
        };

        // The order of the variant types is significant, and should not be changed
        typedef boost::variant<
            // 0x0
            // primary_ns_allocate
            util::tuple<
                naming::gid_type // lower bound
              , naming::gid_type // upper bound
              , boost::uint32_t  // prefix
            >
            // 0x1
            // primary_ns_resolve_gid
            // locality_ns_begin_migration
          , util::tuple<
                naming::gid_type // idbase
              , gva              // gva
              , naming::gid_type // locality
            >
            // 0x2
            // primary_ns_unbind_gid
          , util::tuple<
                gva              // gva
              , naming::gid_type // locality
            >
            // 0x3
            // component_ns_bind_prefix
            // component_ns_bind_name
          , util::tuple<
                components::component_type // ctype
            >
            // 0x4
            // primary_ns_localities
            // component_ns_resolve_id
          , util::tuple<
                std::vector<boost::uint32_t> // prefixes
            >
            // 0x5
            // symbol_ns_unbind
            // symbol_ns_resolve
            // primary_ns_statistics_counter
            // component_ns_statistics_counter
            // symbol_ns_statistics_counter
          , util::tuple<
                naming::gid_type // gid
            >
            // 0x6
            // primary_ns_resolve_locality
          , util::tuple<
                boost::uint32_t // prefix
            >
            // 0x7
            // primary_ns_free
            // primary_ns_bind_gid
            // component_ns_unbind_name
            // component_ns_iterate_types
            // symbol_ns_bind
            // symbol_ns_iterate_names
            // primary_ns_increment_credit
            // primary_ns_decrement_credit
          , util::tuple<
            >
            // 0x8
            // component_ns_get_component_typename
          , util::tuple<
                std::string   // component typename
            >
            // 0x9
            // primary_ns_resolved_localities
          , util::tuple<
                std::map<naming::gid_type, parcelset::endpoints_type>
            >
            // 0xa
            // primary_ns_change_credit_one
          , util::tuple<
                boost::int64_t  // added credits
              , int             // dummy
            >
            // 0xb
            // locality_ns_resolved_locality_gid
          , util::tuple<
                parcelset::endpoints_type  // associated endpoints
            >
        > data_type;

        // {{{ variant helper TODO: consolidate with helpers in request
        template <
            subtype Type
          , int N
        >
        typename hpx::util::tuple_element<
            N, typename boost::mpl::at_c<
                typename data_type::types, Type
            >::type
        >::type
        get_data(
            error_code& ec = throws
            ) const
        { // {{{
            typedef typename boost::mpl::at_c<
                typename data_type::types, Type
            >::type vector_type;

            typedef typename hpx::util::tuple_element<
                N, vector_type
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

                    return hpx::util::get<N>(*v);
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

        data_type data;
    };

    response::response()
      : mc(invalid_request)
      , status(invalid_status)
      , data(new response_data(util::make_tuple()))
    {}

    response::response(
        namespace_action_code type_
      , naming::gid_type lower_
      , naming::gid_type upper_
      , boost::uint32_t prefix_
      , error status_
        )
      : mc(type_)
      , status(status_)
      , data(new response_data(util::make_tuple(lower_, upper_, prefix_)))
    {
        // TODO: verification of namespace_action_code
    }

    response::response(
        namespace_action_code type_
      , naming::gid_type const& gidbase_
      , gva const& gva_
      , naming::gid_type const& locality_
      , error status_
        )
      : mc(type_)
      , status(status_)
      , data(new response_data(util::make_tuple(gidbase_, gva_, locality_)))
    {
        HPX_ASSERT(
            type_ == primary_ns_resolve_gid ||
            type_ == primary_ns_begin_migration);
    }

    response::response(
        namespace_action_code type_
      , gva const& gva_
      , naming::gid_type const& locality_
      , error status_
        )
      : mc(type_)
      , status(status_)
      , data(new response_data(util::make_tuple(gva_, locality_)))
    {
        HPX_ASSERT(type_ == primary_ns_unbind_gid);
    }

    response::response(
        namespace_action_code type_
      , components::component_type ctype_
      , error status_
        )
      : mc(type_)
      , status(status_)
      , data(new response_data(util::make_tuple(ctype_)))
    {
        // TODO: verification of namespace_action_code
    }

    response::response(
        namespace_action_code type_
      , std::vector<boost::uint32_t> const& prefixes_
      , error status_
        )
      : mc(type_)
      , status(status_)
      , data(new response_data(util::make_tuple(prefixes_)))
    {
        // TODO: verification of namespace_action_code
    }

    response::response(
        namespace_action_code type_
      , naming::gid_type gid_
      , error status_
        )
      : mc(type_)
      , status(status_)
      , data(new response_data(util::make_tuple(gid_)))
    {
        // TODO: verification of namespace_action_code
    }

    response::response(
        namespace_action_code type_
      , boost::uint32_t prefix_
      , error status_
        )
      : mc(type_)
      , status(status_)
      , data(new response_data(util::make_tuple(prefix_)))
    {
        // TODO: verification of namespace_action_code
    }

    response::response(
        namespace_action_code type_
      , std::string const& name_
      , error status_
        )
      : mc(type_)
      , status(status_)
      , data(new response_data(util::make_tuple(name_)))
    {
        // TODO: verification of namespace_action_code
    }

    response::response(
        namespace_action_code type_
      , error status_
        )
      : mc(type_)
      , status(status_)
      , data(new response_data(util::make_tuple()))
    {
        // TODO: verification of namespace_action_code
    }

    response::response(
        namespace_action_code type_
      , std::map<naming::gid_type, parcelset::endpoints_type> const & localities_
      , error status_
        )
      : mc(type_)
      , status(status_)
      , data(new response_data(util::make_tuple(localities_)))
    {
        // TODO: verification of namespace_action_code
    }

    response::response(
        namespace_action_code type_
      , boost::int64_t added_credits_
      , error status_
        )
      : mc(type_)
      , status(status_)
      , data(new response_data(util::make_tuple(added_credits_, 0)))
    {
        HPX_ASSERT(
            type_ == primary_ns_increment_credit
         || type_ == primary_ns_decrement_credit);
    }

    response::response(
        namespace_action_code type_
      , parcelset::endpoints_type const & endpoints_
      , error status_
        )
      : mc(type_)
      , status(status_)
      , data(new response_data(util::make_tuple(endpoints_)))
    {
        // TODO: verification of namespace_action_code
    }

    response::~response()
    {}

    response::response(
        response const& other
        )
      : mc(other.mc)
      , status(other.status)
      , data(new response_data(*other.data))
    {}

    response& response::operator=(
        response const& other
        )
    {
        if (this != &other)
        {
            mc = other.mc;
            status = other.status;
            data.reset(new response_data(*other.data));
        }
        return *this;
    }

    gva response::get_gva(
        error_code& ec
        ) const
    {
        switch (data->which())
        {
            case response_data::subtype_gid_gva_prefix:
                return data->get_data<response_data::subtype_gid_gva_prefix, 1>(ec);

            case response_data::subtype_gva:
                return data->get_data<response_data::subtype_gva, 0>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "response::get_gva",
                    "invalid operation for request type");
                return gva();
            }
        }
    }

    std::vector<boost::uint32_t> response::get_localities(
        error_code& ec
        ) const
    {
        return data->get_data<response_data::subtype_prefixes, 0>(ec);
    }

    std::map<naming::gid_type, parcelset::endpoints_type>
    response::get_resolved_localities(
        error_code& ec
        ) const
    {
        return data->get_data<response_data::subtype_resolved_localities, 0>(ec);
    }

    parcelset::endpoints_type
    response::get_endpoints(
        error_code& ec
        ) const
    {
        return data->get_data<response_data::subtype_endpoints, 0>(ec);
    }

    boost::uint32_t response::get_num_localities(
        error_code& ec
        ) const
    {
        return data->get_data<response_data::subtype_prefix, 0>(ec);
    }

    boost::int64_t response::get_added_credits(
        error_code& ec
        ) const
    {
        return data->get_data<response_data::subtype_added_credits, 0>(ec);
    }

    boost::uint32_t response::get_num_overall_threads(
        error_code& ec
        ) const
    {
        std::vector<boost::uint32_t> const& v =
            data->get_data<response_data::subtype_prefixes, 0>(ec);

        return std::accumulate(v.begin(), v.end(), boost::uint32_t(0));
    }

    std::vector<boost::uint32_t> response::get_num_threads(
        error_code& ec
        ) const
    {
        return data->get_data<response_data::subtype_num_threads, 0>(ec);
    }

    components::component_type response::get_component_type(
        error_code& ec
        ) const
    {
        return data->get_data<response_data::subtype_ctype, 0>(ec);
    }

    boost::uint32_t response::get_locality_id(
        error_code& ec
        ) const
    {
        switch (data->which())
        {
            case response_data::subtype_gid_gva_prefix:
                return naming::get_locality_id_from_gid(
                    data->get_data<response_data::subtype_gid_gva_prefix, 2>(ec));

            case response_data::subtype_gid_gid_prefix:
                return data->get_data<response_data::subtype_gid_gid_prefix, 2>(ec);

            case response_data::subtype_prefix:
                return data->get_data<response_data::subtype_prefix, 0>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "response::get_locality_id",
                    "invalid operation for request type");
                return naming::invalid_locality_id;
            }
        }
    }

    naming::gid_type response::get_locality(
        error_code& ec
        ) const
    {
        switch (data->which())
        {
            case response_data::subtype_gid_gva_prefix:
                return data->get_data<response_data::subtype_gid_gva_prefix, 2>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "response::get_locality",
                    "invalid operation for request type");
                return naming::invalid_gid;
            }
        }
    }

    naming::gid_type response::get_base_gid(
        error_code& ec
        ) const
    {
        return data->get_data<response_data::subtype_gid_gva_prefix, 0>(ec);
    }

    naming::gid_type response::get_gid(
        error_code& ec
        ) const
    {
        switch (data->which())
        {
            case response_data::subtype_gid:
                return data->get_data<response_data::subtype_gid, 0>(ec);

            case response_data::subtype_gid_gva_prefix:
                return data->get_data<response_data::subtype_gid_gva_prefix, 2>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "response::get_gid",
                    "invalid operation for response type");
                return naming::invalid_gid;
            }
        }
    }

    naming::gid_type response::get_lower_bound(
        error_code& ec
        ) const
    {
        return data->get_data<response_data::subtype_gid_gid_prefix, 0>(ec);
    }

    naming::gid_type response::get_upper_bound(
        error_code& ec
        ) const
    {
        return data->get_data<response_data::subtype_gid_gid_prefix, 1>(ec);
    }

    std::string response::get_component_typename(
        error_code& ec
        ) const
    {
        return data->get_data<response_data::subtype_string, 0>(ec);
    }

    naming::gid_type response::get_statistics_counter(
        error_code& ec
        ) const
    {
        return data->get_data<response_data::subtype_statistics_counter, 0>(ec);
    }

    struct save_visitor : boost::static_visitor<void>
    {
      private:
        hpx::serialization::output_archive& ar;

      public:
        save_visitor(hpx::serialization::output_archive& ar_)
          : ar(ar_)
        {}

        template <
            typename Sequence
        >
        void operator()(Sequence const& seq) const
        {
            // TODO: verification?
            ar << seq;
        }
    };

    void response::save(serialization::output_archive& ar, const unsigned int) const
    { // {{{
        // TODO: versioning?
        int which = data->which();

        ar & which;
        ar & mc;
        ar & status;
        boost::apply_visitor(save_visitor(ar), data->data);
    } // }}}

#define HPX_LOAD_SEQUENCE(z, n, _)                                          \
    case n:                                                                 \
        {                                                                   \
            boost::mpl::at_c<                                               \
                response_data::data_type::types, n                          \
            >::type d;                                                      \
            ar >> d;                                                        \
            data->data = d;                                                 \
            return;                                                         \
        }                                                                   \
    /**/

    void response::load(serialization::input_archive& ar, const unsigned int)
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
        }
    } // }}}
#undef HPX_LOAD_SEQUENCE
}}

using hpx::lcos::base_lco_with_value;

using hpx::components::component_base_lco_with_value;

using hpx::agas::response;

using hpx::naming::id_type;

HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    hpx::agas::response, hpx_agas_response_type,
    hpx::actions::base_lco_with_value_hpx_agas_response_get,
    hpx::actions::base_lco_with_value_hpx_agas_response_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    std::vector<hpx::agas::response>, hpx_agas_response_vector_type,
    hpx::actions::base_lco_with_value_hpx_agas_response_vector_get,
    hpx::actions::base_lco_with_value_hpx_agas_response_vector_set)

typedef base_lco_with_value<bool, response> base_lco_bool_response_type;
HPX_REGISTER_ACTION_ID(
    base_lco_bool_response_type::set_value_action,
    set_value_action_agas_bool_response_type,
    hpx::actions::set_value_action_agas_bool_response_type_id)

typedef base_lco_with_value<id_type, response> base_lco_id_type_response_type;
HPX_REGISTER_ACTION_ID(
    base_lco_id_type_response_type::set_value_action,
    set_value_action_agas_id_type_response_type,
    hpx::actions::set_value_action_agas_id_type_response_type_id)

