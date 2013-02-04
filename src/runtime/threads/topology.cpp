//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/threads/detail/partlit.hpp>

// #define BOOST_SPIRIT_DEBUG
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

///////////////////////////////////////////////////////////////////////////////
BOOST_FUSION_ADAPT_STRUCT(
    hpx::threads::detail::spec_type,
    (hpx::threads::mask_type, index_min_)
    (hpx::threads::mask_type, index_max_)
    (hpx::threads::detail::spec_type::type, type_)
)

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    //
    //    mappings:
    //        mapping(;mapping)*
    //
    //    mapping:
    //        thread-spec=pu-specs
    //
    //    thread-spec:
    //        t:int
    //        t:int-int
    //        t:all
    //
    //    pu-specs:
    //        pu-spec(.pu-spec)*
    //
    //    pu-spec:
    //        type:int
    //        type:int-int
    //        type:all
    //        ~pu-spec
    //
    //    type:
    //        socket
    //        numanode
    //        core
    //        pu
    //
    namespace qi = boost::spirit::qi;

    // parser for affinity options
    template <typename Iterator>
    struct mappings_parser : qi::grammar<Iterator, mappings_type()>
    {
        mappings_parser()
          : mappings_parser::base_type(start)
        {
            using detail::partlit;
            using detail::spec_type;

            start = mapping % ';';

            mapping =  thread_spec >> '=' >> pu_spec;

            thread_spec =
                    partlit("thread") >> ':'
                    >>  (   qi::uint_
                            >>  ('-' >> qi::uint_ | qi::attr(0ul))
                            >>  qi::attr(spec_type::thread)
                        |   partlit("all") >> qi::attr(~0x0ul) >> qi::attr(0ul)
                            >>  qi::attr(spec_type::thread)
                        )
                ;

            pu_spec =
                    socket_spec >> numanode_spec >> core_spec >> processing_unit_spec
//                 |   '~' >> pu_spec
                ;

            socket_spec =
                    partlit("socket") >> ':'
                    >>  (   qi::uint_
                            >>  ('-' >> qi::uint_ | qi::attr(0ul))
                            >>  qi::attr(spec_type::socket)
                        |   partlit("all") >> qi::attr(~0x0ul) >> qi::attr(0ul)
                            >>  qi::attr(spec_type::socket)
                        )
                |   qi::attr(0ul) >> qi::attr(0ul) >> qi::attr(spec_type::unknown)
                ;

            numanode_spec =
                    -qi::lit('.') >> partlit("numanode") >> ':'
                    >>  (   qi::uint_
                            >>  ('-' >> qi::uint_ | qi::attr(0ul))
                            >>  qi::attr(spec_type::numanode)
                        |   partlit("all") >> qi::attr(~0x0ul) >> qi::attr(0ul)
                            >>  qi::attr(spec_type::numanode)
                        )
                |   qi::attr(0ul) >> qi::attr(0ul) >> qi::attr(spec_type::unknown)
                ;

            core_spec =
                    -qi::lit('.') >> partlit("core") >> ':'
                    >>  (   qi::uint_
                            >>  ('-' >> qi::uint_ | qi::attr(0ul))
                            >>  qi::attr(spec_type::core)
                        |   partlit("all") >> qi::attr(~0x0ul) >> qi::attr(0ul)
                            >>  qi::attr(spec_type::core)
                        )
                |   qi::attr(0ul) >> qi::attr(0ul) >> qi::attr(spec_type::unknown)
                ;

            processing_unit_spec =
                    -qi::lit('.') >> partlit("pu") >> ':'
                    >>  (   qi::uint_
                            >>  ('-' >> qi::uint_ | qi::attr(0ul))
                            >>  qi::attr(spec_type::pu)
                        |   partlit("all") >> qi::attr(~0x0ul) >> qi::attr(0ul)
                            >>  qi::attr(spec_type::pu)
                        )
                |   qi::attr(0ul) >> qi::attr(0ul) >> qi::attr(spec_type::unknown)
                ;

            BOOST_SPIRIT_DEBUG_NODES(
                (start)(mapping)(thread_spec)(pu_spec)(socket_spec)
                (numanode_spec)(core_spec)(processing_unit_spec)
            );
        }

        qi::rule<Iterator, mappings_type()> start;
        qi::rule<Iterator, full_mapping_type()> mapping;
        qi::rule<Iterator, spec_type()> thread_spec;
        qi::rule<Iterator, mapping_type()> pu_spec;
        qi::rule<Iterator, spec_type()> socket_spec;
        qi::rule<Iterator, spec_type()> numanode_spec;
        qi::rule<Iterator, spec_type()> core_spec;
        qi::rule<Iterator, spec_type()> processing_unit_spec;
    };

    template <typename Iterator>
    bool parse(Iterator& begin, Iterator end, mappings_type& m)
    {
        mappings_parser<Iterator> p;
        return qi::parse(begin, end, p, m);
    }

    ///////////////////////////////////////////////////////////////////////////
    void parse_mappings(std::string const& spec, mappings_type& mappings,
        error_code& ec)
    {
        std::string::const_iterator begin = spec.begin();
        if (!detail::parse(begin, spec.end(), mappings) || begin != spec.end())
        {
            HPX_THROWS_IF(ec, bad_parameter, "parse_affinity_options",
                "failed to parse affinity specification: " + spec);
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();
    }

    void decode_mapping(topology const& t, std::size_t thread_num,
        mapping_type const& m, std::vector<mask_type>& affinities,
        error_code& ec)
    {

        if (&ec != &throws)
            ec = make_success_code();
    }

    void decode_mappings(full_mapping_type const& m,
        std::vector<mask_type>& affinities, error_code& ec)
    {
        topology const& t = get_topology();

        // repeat for each of the threads in the affinity specification
        std::size_t first = m.first.index_min_;
        std::size_t last = m.first.index_max_;

        if (first == ~0x0ul) {
            if (last == ~0x0ul) {
                HPX_THROWS_IF(ec, bad_parameter, "decode_mappings",
                    boost::str(boost::format("invalid thread specification, "
                        "min: %x, max %x") % first % last));
                return;
            }

            // bind all threads
            first = 0;
            last = affinities.size();
        }
        if (0 == last)
            last = first;

        for (std::size_t i = first; i != last; ++i)
        {
            decode_mapping(t, i, m.second, affinities, ec);
            if (ec) return;
        }
    }
}}}

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities, error_code& ec)
    {
        detail::mappings_type mappings;
        detail::parse_mappings(spec, mappings, ec);
        if (!ec) {
            BOOST_FOREACH(detail::full_mapping_type const& m, mappings)
            {
                detail::decode_mappings(m, affinities, ec);
                if (ec) return;
            }
        }
    }
}}

