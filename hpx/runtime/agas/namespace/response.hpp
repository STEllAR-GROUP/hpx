////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_FB40C7A4_33B0_4C64_A16B_2A3FEEB237ED)
#define HPX_FB40C7A4_33B0_4C64_A16B_2A3FEEB237ED

#include <limits.h>

#include <boost/move/move.hpp>
#include <boost/assert.hpp>
#include <boost/utility/binary.hpp>
#include <boost/cstdint.hpp>
#include <boost/serialization/split_member.hpp>

#include <hpx/exception.hpp>
#include <hpx/runtime/agas/network/gva.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/component_type.hpp>

namespace hpx { namespace agas
{

template <typename Endpoint>
struct pod_endpoint
{
    typedef pod_endpoint type;
    boost::uint8_t data [sizeof(Endpoint) * CHAR_BIT]; 

    Endpoint& get()
    { return *reinterpret_cast<Endpoint*>(this); } 

    Endpoint const& get() const
    { return *reinterpret_cast<Endpoint const*>(this); } 

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    { ar & get(); }
};

struct pod_gid
{
    boost::uint64_t msb;
    boost::uint64_t lsb;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & msb;
        ar & lsb;
    }
};

template <typename Protocol>
struct pod_gva
{
    typedef typename traits::network::endpoint_type<Protocol>::type
        endpoint_type;

    pod_endpoint<endpoint_type> ep; 
    boost::int32_t ctype;
    boost::uint64_t count;
    boost::uint64_t lva;
    boost::uint64_t offset;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar & ep;
        ar & ctype;
        ar & count;
        ar & lva;
        ar & offset;
    }
};

enum request_code
{ 
    invalid_request             = 0,
    primary_ns_bind_locality    = BOOST_BINARY_U(1000000), 
    primary_ns_bind_gid         = BOOST_BINARY_U(1000001), 
    primary_ns_resolve_locality = BOOST_BINARY_U(1000010), 
    primary_ns_resolve_gid      = BOOST_BINARY_U(1000011), 
    primary_ns_unbind           = BOOST_BINARY_U(1000100), 
    primary_ns_increment        = BOOST_BINARY_U(1000101), 
    primary_ns_decrement        = BOOST_BINARY_U(1000110), 
    primary_ns_localities       = BOOST_BINARY_U(1000111), 
    primary_ns_route            = BOOST_BINARY_U(1001000), 
    component_ns_bind_prefix    = BOOST_BINARY_U(0100000), 
    component_ns_bind_name      = BOOST_BINARY_U(0100001), 
    component_ns_resolve_id     = BOOST_BINARY_U(0100010), 
    component_ns_resolve_name   = BOOST_BINARY_U(0100011), 
    component_ns_unbind         = BOOST_BINARY_U(0100100), 
    symbol_ns_bind              = BOOST_BINARY_U(0010000), 
    symbol_ns_rebind            = BOOST_BINARY_U(0010001), 
    symbol_ns_resolve           = BOOST_BINARY_U(0010010), 
    symbol_ns_unbind            = BOOST_BINARY_U(0010011)  
};

template <typename Protocol>
struct response
{
  private:
    BOOST_COPYABLE_AND_MOVABLE(response);

  public:
    response(void) : rc(invalid_request)
    { clear(); }
    
    response(
        request_code type_
      , error e = success
    ) {
        assign(type_, e);
    }
    
    response(
        request_code type_
      , components::component_type ctype_
      , error e = success
    ) :
      rc(invalid_request)
    {
        assign(type_, ctype_, e);
    }

    response(
        request_code type_
      , boost::int32_t ctype_
      , error e = success
    ) :
      rc(invalid_request)
    {
        assign(type_, ctype_, e);
    }

    response(
        request_code type_
      , naming::gid_type gid_
      , error e = success
    ) :
      rc(invalid_request)
    {
        assign(type_, gid_, e);
    }

    response(
        request_code type_
      , boost::uint32_t prefix_
      , gva<Protocol> const& gva_
      , error e = success
    ) {
        assign(type_, prefix_, gva_, e);
    }

    response(
        request_code type_
      , gva<Protocol> const& gva_
      , error e = success
    ) :
      rc(invalid_request)
    {
        assign(type_, gva_, e);
    }

    response(
        request_code type_
      , boost::uint64_t size_ 
      , boost::uint32_t* array_ 
      , error e = success
    ) {
        assign(type_, size_, array_, e);
    }

    response(
        request_code type_
      , naming::gid_type lower_
      , naming::gid_type upper_
      , boost::uint32_t prefix_
      , error e = success
    ) :
      rc(invalid_request)
    {
        assign(type_, lower_, upper_, prefix_, e);
    }

    response(
        request_code type_
      , boost::uint64_t count_
      , error e = success
    ) :
      rc(invalid_request)
    {
        assign(type_, count_, e);
    }

    response(
        request_code type_
      , boost::uint64_t count_
      , boost::int32_t ctype_
      , error e = success
    ) :
      rc(invalid_request)
    {
        assign(type_, count_, ctype_, e);
    }

    // copy constructor
    response(
        response const& other
    ) :
      rc(invalid_request)
    {
        assign(other);
    }   

    // move constructor
    response(
        BOOST_RV_REF(response) other
    ) :
      rc(invalid_request)
    {
        assign(other);
    }

    ~response()
    { clear(); }

    // copy assignment
    response& operator=(
        BOOST_COPY_ASSIGN_REF(response) other
    ) {
        if (this != &other)
            assign(other); 
        return *this;
    }

    // move assignment
    response& operator=(
        BOOST_RV_REF(response) other
    ) {
        if (this != &other)
            assign(other);
        return *this;
    }

    // copy assignment (implementation)
    void assign(
        BOOST_COPY_ASSIGN_REF(response) other
    ) { // {{{
        clear();

        rc = other.rc;
        status = other.status;

        switch (rc) {
            default:
                return;

            case component_ns_bind_prefix:
            case component_ns_bind_name:
            case component_ns_resolve_name: {
                data.ctype = other.data.ctype;
                return;
            }

            case symbol_ns_rebind:
            case symbol_ns_resolve: {
                data.gid.msb = other.data.gid.msb;
                data.gid.lsb = other.data.gid.lsb;
                return;
            }

            case primary_ns_resolve_locality: {
                data.resolved_locality.prefix
                    = other.data.resolved_locality.prefix;
                data.resolved_locality.gva.ep.get()
                    = other.data.resolved_locality.gva.ep.get();
                data.resolved_locality.gva.ctype
                    = other.data.resolved_locality.gva.ctype;
                data.resolved_locality.gva.count
                    = other.data.resolved_locality.gva.count;
                data.resolved_locality.gva.lva
                    = other.data.resolved_locality.gva.lva;
                data.resolved_locality.gva.offset
                    = other.data.resolved_locality.gva.offset;
                return;
            }

            case primary_ns_resolve_gid:
            case primary_ns_unbind: {
                data.gva.ep.get() = other.data.gva.ep.get();
                data.gva.ctype = other.data.gva.ctype;
                data.gva.count = other.data.gva.count;
                data.gva.lva = other.data.gva.lva;
                data.gva.offset = other.data.gva.offset;
                return;
            }

            case primary_ns_localities:
            case component_ns_resolve_id: {
                data.localities.size = other.data.localities.size;

                if (data.localities.size > 0)
                {
                    data.localities.array
                        = new boost::uint32_t [data.localities.size];

                    for (boost::uint64_t i = 0; i < data.localities.size; ++i)
                        data.localities.array[i]
                            = other.data.localities.array[i];
                }

                else {
                    BOOST_ASSERT(other.data.localities.array == 0);
                    data.localities.array = 0;
                }

                return;
            }

            case primary_ns_bind_locality: {
                data.locality_binding.lower.msb
                    = other.data.locality_binding.lower.msb;
                data.locality_binding.lower.lsb
                    = other.data.locality_binding.lower.lsb;
                data.locality_binding.upper.msb
                    = other.data.locality_binding.upper.msb;
                data.locality_binding.upper.lsb
                    = other.data.locality_binding.upper.lsb;
                data.locality_binding.prefix
                    = other.data.locality_binding.prefix;
                return;
            }

            case primary_ns_increment: {
                data.count = other.data.count;
                return;
            }

            case primary_ns_decrement: {
                data.decrement.count = other.data.decrement.count;
                data.decrement.ctype = other.data.decrement.ctype;
                return;
            }
        };
    } // }}}

    // move assignment (implementation)
    void assign(
        BOOST_RV_REF(response) other
    ) { // {{{
        clear();

        rc = other.rc;
        status = other.status;

        switch (rc) {
            default:
                break;

            case component_ns_bind_prefix:
            case component_ns_bind_name:
            case component_ns_resolve_name: {
                data.ctype = other.data.ctype;
                break;
            }

            case symbol_ns_rebind:
            case symbol_ns_resolve: {
                data.gid.msb = other.data.gid.msb;
                data.gid.lsb = other.data.gid.lsb;
                break;
            }

            case primary_ns_resolve_locality: {
                data.resolved_locality.prefix
                    = other.data.resolved_locality.prefix;
                data.resolved_locality.gva.ep.get()
                    = other.data.resolved_locality.gva.ep.get();
                data.resolved_locality.gva.ctype
                    = other.data.resolved_locality.gva.ctype;
                data.resolved_locality.gva.count
                    = other.data.resolved_locality.gva.count;
                data.resolved_locality.gva.lva
                    = other.data.resolved_locality.gva.lva;
                data.resolved_locality.gva.offset
                    = other.data.resolved_locality.gva.offset;
                break;
            }

            case primary_ns_resolve_gid:
            case primary_ns_unbind: {
                data.gva.ep.get() = other.data.gva.ep.get();
                data.gva.ctype = other.data.gva.ctype;
                data.gva.count = other.data.gva.count;
                data.gva.lva = other.data.gva.lva;
                data.gva.offset = other.data.gva.offset;
                break;
            }

            case primary_ns_localities:
            case component_ns_resolve_id: {
                data.localities.size = other.data.localities.size;
                data.localities.array = other.data.localities.array;
                break;
            }

            case primary_ns_bind_locality: {
                data.locality_binding.lower.msb
                    = other.data.locality_binding.lower.msb;
                data.locality_binding.lower.lsb
                    = other.data.locality_binding.lower.lsb;
                data.locality_binding.upper.msb
                    = other.data.locality_binding.upper.msb;
                data.locality_binding.upper.lsb
                    = other.data.locality_binding.upper.lsb;
                data.locality_binding.prefix
                    = other.data.locality_binding.prefix;
                return;
            }

            case primary_ns_increment: {
                data.count = other.data.count;
                break;
            }

            case primary_ns_decrement: {
                data.decrement.count = other.data.decrement.count;
                data.decrement.ctype = other.data.decrement.ctype;
                break;
            }
        };

        other.rc = invalid_request; // prevent deallocation
        other.clear();
    } // }}}

    // primary_ns_bind_gid
    // primary_ns_route
    // component_ns_unbind
    // symbol_ns_bind
    // symbol_ns_unbind
    void assign(
        request_code type_
      , error e = success
    ) { // {{{
        clear();

        switch (type_)
        {
            case primary_ns_bind_gid:
            case primary_ns_route:
            case component_ns_unbind:
            case symbol_ns_bind:
            case symbol_ns_unbind:
                break;

            default: {
                HPX_THROW_EXCEPTION(bad_response_type, 
                    "response::assign", "invalid response created");
            }
        };

        rc = type_;
        status = e;
    } // }}} 

    // forwarder
    void assign(
        request_code type_
      , components::component_type ctype_
      , error e = success
    ) {
        assign(type_, boost::int32_t(ctype_), e);
    }

    // component_ns_bind_prefix
    // component_ns_bind_name
    // component_ns_resolve_name
    void assign(
        request_code type_
      , boost::int32_t ctype_
      , error e = success
    ) { // {{{
        clear();

        switch (type_)
        {
            case component_ns_bind_prefix:
            case component_ns_bind_name:
            case component_ns_resolve_name:
                break;

            default: {
                HPX_THROW_EXCEPTION(bad_response_type, 
                    "response::assign", "invalid response created");
            }
        };

        data.ctype = ctype_;
        rc = type_;
        status = e;
    } // }}} 
   
    // symbol_ns_rebind
    // symbol_ns_resolve 
    void assign(
        request_code type_
      , naming::gid_type gid_
      , error e = success
    ) { // {{{
        clear();

        switch (type_)
        {
            case symbol_ns_rebind:
            case symbol_ns_resolve:
                break;

            default: {
                HPX_THROW_EXCEPTION(bad_response_type, 
                    "response::assign", "invalid response created");
            }
        };

        data.gid.msb = gid_.get_msb();
        data.gid.lsb = gid_.get_lsb();
        rc = type_;
        status = e;
    } // }}} 
    
    // primary_ns_resolve_locality
    void assign(
        request_code type_
      , boost::uint32_t prefix_
      , gva<Protocol> const& gva_
      , error e = success
    ) { // {{{
        clear();

        switch (type_)
        {
            case primary_ns_resolve_locality:
                break;

            default: {
                HPX_THROW_EXCEPTION(bad_response_type, 
                    "response::assign", "invalid response created");
            }
        };

        data.resolved_locality.prefix = prefix_;
        data.resolved_locality.gva.ep.get() = gva_.endpoint;
        data.resolved_locality.gva.ctype = gva_.type;
        data.resolved_locality.gva.count = gva_.count;
        data.resolved_locality.gva.lva = gva_.lva();
        data.resolved_locality.gva.offset = gva_.offset;
        rc = type_;
        status = e;
    } // }}} 

    // primary_ns_resolve_gid
    // primary_ns_unbind
    void assign(
        request_code type_
      , gva<Protocol> const& gva_
      , error e = success
    ) { // {{{
        clear();

        switch (type_)
        {
            case primary_ns_resolve_gid:
            case primary_ns_unbind:
                break;

            default: {
                HPX_THROW_EXCEPTION(bad_response_type, 
                    "response::assign", "invalid response created");
            }
        };

        data.gva.ep.get() = gva_.endpoint;
        data.gva.ctype = gva_.type;
        data.gva.count = gva_.count;
        data.gva.lva = gva_.lva();
        data.gva.offset = gva_.offset;
        rc = type_;
        status = e;
    } // }}} 

    // primary_ns_localities
    // component_ns_resolve_id
    void assign(
        request_code type_
      , boost::uint64_t size_ 
      , boost::uint32_t* array_ 
      , error e = success
    ) { // {{{
        clear();

        switch (type_)
        {
            case primary_ns_localities:
            case component_ns_resolve_id:
                break;

            default: {
                HPX_THROW_EXCEPTION(bad_response_type, 
                    "response::assign", "invalid response created");
            }
        };

        if (size_ == 0)
            BOOST_ASSERT(array_ == 0);

        data.localities.size = size_;
        data.localities.array = array_;
        rc = type_;
        status = e;
    } // }}} 

    // primary_ns_bind_locality
    void assign(
        request_code type_
      , naming::gid_type lower_
      , naming::gid_type upper_
      , boost::uint32_t prefix_
      , error e = success
    ) { // {{{
        clear();

        switch (type_)
        {
            case primary_ns_bind_locality:
                break;

            default: {
                HPX_THROW_EXCEPTION(bad_response_type, 
                    "response::assign", "invalid response created");
            }
        };

        data.locality_binding.lower.msb = lower_.get_msb();
        data.locality_binding.lower.lsb = lower_.get_lsb();
        data.locality_binding.upper.msb = upper_.get_msb();
        data.locality_binding.upper.lsb = upper_.get_lsb();
        data.locality_binding.prefix = prefix_;
        rc = type_;
        status = e;
    } // }}} 

    // primary_ns_increment
    void assign(
        request_code type_
      , boost::uint64_t count_
      , error e = success
    ) { // {{{
        clear();

        switch (type_)
        {
            case primary_ns_increment:
                break;

            default: {
                HPX_THROW_EXCEPTION(bad_response_type, 
                    "response::assign", "invalid response created");
            }
        };

        data.count = count_;
        rc = type_;
        status = e;
    } // }}} 

    // forwarder
    void assign(
        request_code type_
      , boost::uint64_t count_
      , components::component_type ctype_
      , error e = success
    ) {
        assign(type_, count_, boost::int32_t(ctype_), e);
    }

    // primary_ns_decrement
    void assign(
        request_code type_
      , boost::uint64_t count_
      , boost::int32_t ctype_
      , error e = success
    ) { // {{{
        clear();

        switch (type_)
        {
            case primary_ns_decrement:
                break;

            default: {
                HPX_THROW_EXCEPTION(bad_response_type, 
                    "response::assign", "invalid response created");
            }
        };

        data.decrement.count = count_;
        data.decrement.ctype = ctype_;
        rc = type_;
        status = e;
    } // }}} 

    gva<Protocol> get_gva() const
    { // {{{
        switch (rc)
        {
            case primary_ns_unbind:
            case primary_ns_resolve_gid:
                return gva<Protocol>(data.gva.ep.get()
                                   , data.gva.ctype
                                   , data.gva.count
                                   , data.gva.lva
                                   , data.gva.offset);
            case primary_ns_resolve_locality:
                return gva<Protocol>(data.resolved_locality.gva.ep.get()
                                   , data.resolved_locality.gva.ctype
                                   , data.resolved_locality.gva.count
                                   , data.resolved_locality.gva.lva
                                   , data.resolved_locality.gva.offset);

            default: {
                HPX_THROW_EXCEPTION(bad_parameter, 
                    "response::get_gva",
                    "invalid operation for request type");
                return gva<Protocol>();
            }
        };
    } // }}}

    boost::uint64_t get_count() const
    { // {{{
        switch (rc)
        {
            case primary_ns_increment:
                return data.count;

            case primary_ns_decrement:
                return data.decrement.count;

            default: {
                HPX_THROW_EXCEPTION(bad_parameter, 
                    "response::get_count",
                    "invalid operation for request type");
                return 0;
            }
        };
    } // }}} 

    boost::uint32_t* get_localities() const
    { // {{{
        switch (rc)
        {
            case primary_ns_localities:
            case component_ns_resolve_id:
                return data.localities.array;

            default: {
                HPX_THROW_EXCEPTION(bad_parameter, 
                    "response::get_localities",
                    "invalid operation for request type");
                return 0;
            }
        };
    } // }}} 

    boost::uint64_t get_localities_size() const
    { // {{{
        switch (rc)
        {
            case primary_ns_localities:
            case component_ns_resolve_id:
                return data.localities.size;

            default: {
                HPX_THROW_EXCEPTION(bad_parameter, 
                    "response::get_localities_size",
                    "invalid operation for request type");
                return 0;
            }
        };
    } // }}} 

    boost::int32_t get_component_type() const
    { // {{{
        switch (rc)
        {
            case primary_ns_decrement:
                return data.decrement.ctype;

            case component_ns_bind_prefix:
            case component_ns_bind_name:
            case component_ns_resolve_name:
                return data.ctype;

            default: {
                HPX_THROW_EXCEPTION(bad_parameter, 
                    "response::get_component_type",
                    "invalid operation for request type");
                return 0;
            }
        };
    } // }}} 

    boost::uint32_t get_prefix() const
    { // {{{
        switch (rc)
        {
            case primary_ns_resolve_locality:
                return data.resolved_locality.prefix;

            case primary_ns_bind_locality:
                return data.locality_binding.prefix;

            default: {
                HPX_THROW_EXCEPTION(bad_parameter, 
                    "response::get_component_type",
                    "invalid operation for request type");
                return 0;
            }
        };
    } // }}} 

    naming::gid_type get_gid() const
    { // {{{
        switch (rc)
        {
            case symbol_ns_rebind:
            case symbol_ns_resolve:
                return naming::gid_type(data.gid.msb, data.gid.lsb);

            default: {
                HPX_THROW_EXCEPTION(bad_parameter, 
                    "response::get_gid",
                    "invalid operation for request type");
                return naming::invalid_gid;
            }
        };
    } // }}} 

    naming::gid_type get_lower_bound() const
    { // {{{
        switch (rc)
        {
            case primary_ns_bind_locality:
                return naming::gid_type(data.locality_binding.lower.msb
                                      , data.locality_binding.lower.lsb);

            default: {
                HPX_THROW_EXCEPTION(bad_parameter, 
                    "response::get_lower_bound",
                    "invalid operation for request type");
                return naming::invalid_gid;
            }
        };
    } // }}} 

    naming::gid_type get_upper_bound() const
    { // {{{
        switch (rc)
        {
            case primary_ns_bind_locality:
                return naming::gid_type(data.locality_binding.upper.msb
                                      , data.locality_binding.upper.lsb);

            default: {
                HPX_THROW_EXCEPTION(bad_parameter, 
                    "response::get_upper_bound",
                    "invalid operation for request type");
                return naming::invalid_gid;
            }
        };
    } // }}} 

    error get_status() const
    { return status; }

    void clear()
    { // {{{
        switch (rc)
        {
            case primary_ns_localities:
            case component_ns_resolve_id: {
                if (data.localities.array)
                {
                    BOOST_ASSERT(data.localities.size != 0);
                    delete[] data.localities.array;
                    break;
                } 
            }
            default:
                break;
        };
 
        typedef typename boost::uint_t<
            boost::alignment_of<data_type>::value * CHAR_BIT
        >::exact unit_type;

        unit_type* p = reinterpret_cast<unit_type*>(&data);

        for (std::size_t i = 0, end = sizeof(data_type) / sizeof(unit_type);
             i != end; ++i)
            p[i] = 0;

        status = success;
        rc = invalid_request;
    } // }}}

    request_code which() const
    { return rc; } 
        
  private:
    friend class boost::serialization::access;

    template <typename Archive>
    void save(
        Archive& ar
      , const unsigned int
    ) const { // {{{
        ar & rc; 
        ar & status;

        switch (rc) {
            default:
                return;

            case component_ns_bind_prefix:
            case component_ns_bind_name:
            case component_ns_resolve_name: {
                ar & data.ctype;
                return;
            }

            case symbol_ns_rebind:
            case symbol_ns_resolve: {
                ar & data.gid.msb;
                ar & data.gid.lsb;
                return;
            }

            case primary_ns_resolve_locality: {
                ar & data.resolved_locality.prefix;
                ar & data.resolved_locality.gva.ep.get();
                ar & data.resolved_locality.gva.ctype;
                ar & data.resolved_locality.gva.count;
                ar & data.resolved_locality.gva.lva;
                ar & data.resolved_locality.gva.offset;
                return;
            }

            case primary_ns_resolve_gid:
            case primary_ns_unbind: {
                ar & data.gva.ep.get();
                ar & data.gva.ctype;
                ar & data.gva.count;
                ar & data.gva.lva;
                ar & data.gva.offset;
                return;
            }

            case primary_ns_localities:
            case component_ns_resolve_id: {
                ar & data.localities.size;

                if (data.localities.size > 0)
                {
                    for (boost::uint64_t i = 0; i < data.localities.size; ++i)
                        ar & data.localities.array[i];
                }

                return;
            }

            case primary_ns_bind_locality: {
                ar & data.locality_binding.lower.msb;
                ar & data.locality_binding.lower.lsb;
                ar & data.locality_binding.upper.msb;
                ar & data.locality_binding.upper.lsb;
                ar & data.locality_binding.prefix;
                return;
            }

            case primary_ns_increment: {
                ar & data.count;
                return;
            }

            case primary_ns_decrement: {
                ar & data.decrement.count;
                ar & data.decrement.ctype;
                return;
            }
        };
    } // }}}

    template <typename Archive>
    void load(
        Archive& ar
      , const unsigned int
    ) { // {{{
        clear();

        ar & rc;
        ar & status;

        switch (rc) {
            default:
                return;

            case component_ns_bind_prefix:
            case component_ns_bind_name:
            case component_ns_resolve_name: {
                ar & data.ctype;
                return;
            }

            case symbol_ns_rebind:
            case symbol_ns_resolve: {
                ar & data.gid.msb;
                ar & data.gid.lsb;
                return;
            }

            case primary_ns_resolve_locality: {
                ar & data.resolved_locality.prefix;
                ar & data.resolved_locality.gva.ep.get();
                ar & data.resolved_locality.gva.ctype;
                ar & data.resolved_locality.gva.count;
                ar & data.resolved_locality.gva.lva;
                ar & data.resolved_locality.gva.offset;
                return;
            }

            case primary_ns_resolve_gid:
            case primary_ns_unbind: {
                ar & data.gva.ep.get();
                ar & data.gva.ctype;
                ar & data.gva.count;
                ar & data.gva.lva;
                ar & data.gva.offset;
                return;
            }

            case primary_ns_localities:
            case component_ns_resolve_id: {
                ar & data.localities.size;

                if (data.localities.size > 0)
                {
                    data.localities.array
                        = new boost::uint32_t [data.localities.size];

                    for (boost::uint64_t i = 0; i < data.localities.size; ++i)
                        ar & data.localities.array[i];
                }

                else
                    data.localities.array = 0; 

                return;
            }

            case primary_ns_bind_locality: {
                ar & data.locality_binding.lower.msb;
                ar & data.locality_binding.lower.lsb;
                ar & data.locality_binding.upper.msb;
                ar & data.locality_binding.upper.lsb;
                ar & data.locality_binding.prefix;
                return;
            }

            case primary_ns_increment: {
                ar & data.count;
                return;
            }

            case primary_ns_decrement: {
                ar & data.decrement.count;
                ar & data.decrement.ctype;
                return;
            }
        };
    } // }}}

    BOOST_SERIALIZATION_SPLIT_MEMBER()

    request_code rc;
    error status;

    union data_type
    {
        struct locality_binding_type
        {
            pod_gid lower;
            pod_gid upper;
            boost::uint32_t prefix;
        } locality_binding;

        struct resolved_locality_type
        {
            boost::uint32_t prefix;
            pod_gva<Protocol> gva;
        } resolved_locality;

        pod_gva<Protocol> gva;

        boost::uint64_t count;

        struct decrement_type
        {
            boost::uint64_t count;
            boost::int32_t ctype; 
        } decrement;

        struct localities_type
        {
            boost::uint64_t size;
            boost::uint32_t* array;
        } localities; 

        boost::int32_t ctype;

        pod_gid gid;
    } data;
};

}}

#endif // HPX_FB40C7A4_33B0_4C64_A16B_2A3FEEB237ED

