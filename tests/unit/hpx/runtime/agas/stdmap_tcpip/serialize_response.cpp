////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <vector>

#include <boost/iostreams/stream.hpp>
#include <boost/archive/basic_binary_iarchive.hpp>
#include <boost/archive/basic_binary_oarchive.hpp>

#include <hpx/util/container_device.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/runtime/agas/network/backend/tcpip.hpp>
#include <hpx/runtime/agas/namespace/response.hpp>
#include <hpx/util/lightweight_test.hpp>

typedef hpx::agas::response<
    hpx::agas::tag::network::tcpip
> response_type;

typedef hpx::agas::gva<
    hpx::agas::tag::network::tcpip
> gva_type;

using hpx::success;
using hpx::no_success;

using hpx::components::component_base_lco;
using hpx::components::component_memory_block;
using hpx::components::component_dataflow_block;

using hpx::naming::gid_type;

using hpx::agas::invalid_request;
using hpx::agas::primary_ns_bind_locality;
using hpx::agas::primary_ns_bind_gid;
using hpx::agas::primary_ns_resolve_locality;
using hpx::agas::primary_ns_resolve_gid;
using hpx::agas::primary_ns_unbind;
using hpx::agas::primary_ns_increment;
using hpx::agas::primary_ns_decrement;
using hpx::agas::primary_ns_localities;
using hpx::agas::component_ns_bind_prefix;
using hpx::agas::component_ns_bind_name;
using hpx::agas::component_ns_resolve_id;
using hpx::agas::component_ns_resolve_name;
using hpx::agas::component_ns_unbind;
using hpx::agas::symbol_ns_bind;
using hpx::agas::symbol_ns_rebind;
using hpx::agas::symbol_ns_resolve;
using hpx::agas::symbol_ns_unbind;

int main()
{
    typedef std::vector<char> buffer_type;
    typedef hpx::util::container_device<buffer_type> device_type;
    typedef boost::iostreams::stream<device_type> stream_type;

    #if HPX_USE_PORTABLE_ARCHIVES != 0
        typedef hpx::util::portable_binary_oarchive oarchive_type;
        typedef hpx::util::portable_binary_iarchive iarchive_type;
    #else
        typedef boost::archive::binary_oarchive oarchive_type;
        typedef boost::archive::binary_iarchive iarchive_type;
    #endif
    
    {   // primary_ns_bind_gid
        // component_ns_unbind
        // symbol_ns_bind
        // symbol_ns_unbind
        buffer_type buffer; 

        response_type r0(primary_ns_bind_gid, success)
                    , r1(primary_ns_bind_gid, no_success)
                    , r2(component_ns_unbind, success)
                    , r3(component_ns_unbind, no_success)
                    , r4(symbol_ns_bind,      success)
                    , r5(symbol_ns_bind,      no_success)
                    , r6(symbol_ns_unbind,    success)
                    , r7(symbol_ns_unbind,    no_success);
    
        { // save 
            stream_type stream(buffer);
            oarchive_type archive(stream);

            archive & r0;
            archive & r1;
            archive & r2;
            archive & r3;
            archive & r4;
            archive & r5;
            archive & r6;
            archive & r7;
        }
    
        { // load
            stream_type stream(buffer);
            iarchive_type archive(stream);
            response_type l0, l1, l2, l3, l4, l5, l6, l7;

            archive & l0;
            archive & l1;
            archive & l2;
            archive & l3;
            archive & l4;
            archive & l5;
            archive & l6;
            archive & l7;
    
            HPX_TEST_EQ(unsigned(r0.which()),      unsigned(l0.which()));
            HPX_TEST_EQ(unsigned(r0.get_status()), unsigned(l0.get_status()));
            HPX_TEST_EQ(unsigned(r1.which()),      unsigned(l1.which()));
            HPX_TEST_EQ(unsigned(r1.get_status()), unsigned(l1.get_status()));
            HPX_TEST_EQ(unsigned(r2.which()),      unsigned(l2.which()));
            HPX_TEST_EQ(unsigned(r2.get_status()), unsigned(l2.get_status()));
            HPX_TEST_EQ(unsigned(r3.which()),      unsigned(l3.which()));
            HPX_TEST_EQ(unsigned(r3.get_status()), unsigned(l3.get_status()));
            HPX_TEST_EQ(unsigned(r4.which()),      unsigned(l4.which()));
            HPX_TEST_EQ(unsigned(r4.get_status()), unsigned(l4.get_status()));
            HPX_TEST_EQ(unsigned(r5.which()),      unsigned(l5.which()));
            HPX_TEST_EQ(unsigned(r5.get_status()), unsigned(l5.get_status()));
            HPX_TEST_EQ(unsigned(r6.which()),      unsigned(l6.which()));
            HPX_TEST_EQ(unsigned(r6.get_status()), unsigned(l6.get_status()));
            HPX_TEST_EQ(unsigned(r7.which()),      unsigned(l7.which()));
            HPX_TEST_EQ(unsigned(r7.get_status()), unsigned(l7.get_status()));
        }
    }

    return hpx::util::report_errors();
}

