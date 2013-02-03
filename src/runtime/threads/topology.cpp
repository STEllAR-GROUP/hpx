//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/threads/detail/partlit.hpp>

//#define BOOST_SPIRIT_DEBUG
#define BOOST_SPIRIT_USE_PHOENIX_V3
#include <boost/spirit/include/qi_char.hpp>
#include <boost/spirit/include/qi_nonterminal.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/qi_operator.hpp>
#include <boost/spirit/include/qi_parse.hpp>
#include <boost/spirit/include/qi_string.hpp>
#include <boost/spirit/include/qi_auxiliary.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>

#include <boost/foreach.hpp>

#if defined(__ANDROID__) && defined(ANDROID)
#include <cpu-features.h>
#endif

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    mask_type topology::get_service_affinity_mask(
        mask_type used_processing_units, error_code& ec) const
    {
        // We bind the service threads to the first numa domain. This is useful
        // as the first numa domain is likely to have the PCI controllers etc.
        mask_type machine_mask = this->get_numa_node_affinity_mask(0, true, ec);
        if (ec || 0 == machine_mask)
            return 0;

        if (&ec != &throws)
            ec = make_success_code();

        mask_type res = ~used_processing_units & machine_mask;

        if(res == 0) return machine_mask;
        else return res;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t hardware_concurrency()
    {
#if defined(__ANDROID__) && defined(ANDROID)
        static std::size_t num_of_cores = ::android_getCpuCount();
#else
        static std::size_t num_of_cores = boost::thread::hardware_concurrency();
#endif

        if (0 == num_of_cores)
            return 1;           // Assume one core.

        return num_of_cores;
    }

    topology const& get_topology()
    {
        return get_runtime().get_topology();
    }
}}

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    //
    // mappings:
    //     mapping(;mapping)*
    //
    // mapping:
    //     thread-spec=pu-specs
    //
    // thread-spec:
    //     t:int
    //     t:int-int
    //     t:all
    //
    // pu-specs:
    //     pu-spec(.pu-spec)*
    //
    // pu-spec:
    //     type:int
    //     type:int-int
    //     type:all
    //     ~mapping
    //
    // type:
    //     socket
    //     numanode
    //     core
    //     pu

    struct spec_type
    {
        enum type { unknown, thread, socket, numanode, core, pu };

        spec_type()
          : type_(unknown), index_min_(0), index_max_(0)
        {}

        spec_type(type t)
          : type_(t), index_min_(0), index_max_(0)
        {}

        type type_;
        mask_type index_min_;
        mask_type index_max_;
    };

#if defined(BOOST_SPIRIT_DEBUG)
    std::ostream& operator<<(std::ostream& os, spec_type const& spec)
    {
        return os;
    }
#endif

    struct thread_spec_type : spec_type
    {
        thread_spec_type() : spec_type(spec_type::thread) {}
    };

    struct socket_spec_type : spec_type
    {
        socket_spec_type() : spec_type(spec_type::socket) {}
    };

    struct numanode_spec_type : spec_type
    {
        numanode_spec_type() : spec_type(spec_type::numanode) {}
    };

    struct core_spec_type : spec_type
    {
        core_spec_type() : spec_type(spec_type::core) {}
    };

    struct pu_spec_type : spec_type
    {
        pu_spec_type() : spec_type(spec_type::pu) {}
    };
}}}

BOOST_FUSION_ADAPT_STRUCT(
    hpx::threads::detail::thread_spec_type,
    (hpx::threads::mask_type, index_min_)
    (hpx::threads::mask_type, index_max_)
)
BOOST_FUSION_ADAPT_STRUCT(
    hpx::threads::detail::socket_spec_type,
    (hpx::threads::mask_type, index_min_)
    (hpx::threads::mask_type, index_max_)
)
BOOST_FUSION_ADAPT_STRUCT(
    hpx::threads::detail::numanode_spec_type,
    (hpx::threads::mask_type, index_min_)
    (hpx::threads::mask_type, index_max_)
)
BOOST_FUSION_ADAPT_STRUCT(
    hpx::threads::detail::core_spec_type,
    (hpx::threads::mask_type, index_min_)
    (hpx::threads::mask_type, index_max_)
)
BOOST_FUSION_ADAPT_STRUCT(
    hpx::threads::detail::pu_spec_type,
    (hpx::threads::mask_type, index_min_)
    (hpx::threads::mask_type, index_max_)
)

namespace hpx { namespace threads { namespace detail
{
    typedef std::vector<spec_type> mapping_type;
    typedef std::pair<thread_spec_type, mapping_type> full_mapping_type;
    typedef std::vector<full_mapping_type> mappings_type;

    namespace qi = boost::spirit::qi;

    // We need this wrapper as spirit::attr_casthas a bug and can't be used 
    // with rules.
    template <typename Exposed, typename OutputIterator,
        typename T1, typename T2, typename T3, typename T4>
    boost::spirit::stateful_tag_type<
        typename qi::rule<OutputIterator, T1, T2, T3, T4>::reference_,
        boost::spirit::tag::attr_cast,
        Exposed
    >
    attr_cast(qi::rule<OutputIterator, T1, T2, T3, T4> const& r)
    {
        return boost::spirit::stateful_tag_type<
            typename qi::rule<OutputIterator, T1, T2, T3, T4>::reference_,
            boost::spirit::tag::attr_cast,
            Exposed
        >(r.alias());
    }

    // parser for affinity options
    template <typename Iterator>
    struct mappings_parser : qi::grammar<Iterator, mappings_type()>
    {
        mappings_parser()
          : mappings_parser::base_type(start)
        {
            using detail::partlit;

            start = mapping % ';';

            mapping =  thread_spec >> '=' >> pu_spec;

            thread_spec =
                    partlit("thread") >> ':'
                    >>  (   qi::int_ >> -('-' >> qi::int_)
                        |   partlit("all") >> qi::attr(~0x0) >> qi::attr(0)
                        )
                ;

            pu_spec =
                    attr_cast<spec_type>(socket_spec)
                    >>  attr_cast<spec_type>(numanode_spec)
                    >>  attr_cast<spec_type>(core_spec)
                    >>  attr_cast<spec_type>(processing_unit_spec)
                |   '~' >> pu_spec
                ;

            socket_spec =
                    partlit("socket") >> ':'
                    >>  (   qi::int_ >> -('-' >> qi::int_)
                        |   partlit("all") >> qi::attr(~0x0) >> qi::attr(0)
                        )
                |   qi::attr(0) >> qi::attr(0)
                ;

            numanode_spec =
                    -qi::lit('.') >> partlit("numanode") >> ':'
                    >>  (   qi::int_ >> -('-' >> qi::int_)
                        |   partlit("all") >> qi::attr(~0x0) >> qi::attr(0)
                        )
                |   qi::attr(0) >> qi::attr(0)
                ;

            core_spec =
                    -qi::lit('.') >> partlit("core") >> ':'
                    >>  (   qi::int_ >> -('-' >> qi::int_)
                        |   partlit("all") >> qi::attr(~0x0) >> qi::attr(0)
                        )
                |   qi::attr(0) >> qi::attr(0)
                ;

            processing_unit_spec =
                    -qi::lit('.') >> partlit("pu") >> ':'
                    >>  (   qi::int_ >> -('-' >> qi::int_)
                        |   partlit("all") >> qi::attr(~0x0) >> qi::attr(0)
                        )
                |   qi::attr(0) >> qi::attr(0)
                ;

            BOOST_SPIRIT_DEBUG_NODES(
                (start)(mapping)(thread_spec)(pu_spec)(socket_spec)
                (numanode_spec)(core_spec)(processing_unit_spec)
            );
        }

        qi::rule<Iterator, mappings_type()> start;
        qi::rule<Iterator, full_mapping_type()> mapping;
        qi::rule<Iterator, thread_spec_type()> thread_spec;
        qi::rule<Iterator, mapping_type()> pu_spec;
        qi::rule<Iterator, socket_spec_type()> socket_spec;
        qi::rule<Iterator, numanode_spec_type()> numanode_spec;
        qi::rule<Iterator, core_spec_type()> core_spec;
        qi::rule<Iterator, pu_spec_type()> processing_unit_spec;
    };

    template <typename Iterator>
    bool parse(Iterator& begin, Iterator end, mappings_type& m)
    {
        mappings_parser<Iterator> p;
        return qi::parse(begin, end, p, m);
    }

    ///////////////////////////////////////////////////////////////////////////
    bool decode_mapping(full_mapping_type const& m,
        std::vector<mask_type>& affinities)
    {
        return true;
    }
}}}

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    bool parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities)
    {
        detail::mappings_type mappings;
        std::string::const_iterator begin = spec.begin();
        if (!detail::parse(begin, spec.end(), mappings) || begin != spec.end())
            return false;

        BOOST_FOREACH(detail::full_mapping_type const& m, mappings)
        {
            if (!detail::decode_mapping(m, affinities))
                return false;
        }

        return true;
    }
}}

