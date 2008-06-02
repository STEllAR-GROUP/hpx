//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_CONFIG_MAR_24_2008_0943AM)
#define HPX_CONFIG_MAR_24_2008_0943AM

///////////////////////////////////////////////////////////////////////////////
/// This is the default ip/port number used by the global address resolver
#define HPX_PORT 7810
#define HPX_NAME_RESOLVER_PORT 7911

///////////////////////////////////////////////////////////////////////////////
/// This defines if the Intel Thread Building Blocks library will be used
#if !defined(HPX_USE_TBB)
#define HPX_USE_TBB 0
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the maximum number of constructor arguments a component 
/// can take
#if !defined(HPX_COMPONENT_ARGUMENT_LIMIT)
#define HPX_COMPONENT_ARGUMENT_LIMIT 6
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the maximum number of arguments an action can take
#if !defined(HPX_ACTION_ARGUMENT_LIMIT)
#define HPX_ACTION_ARGUMENT_LIMIT 6
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the number kept alive of outgoing (parcel-) connections
#if !defined(HPX_MAX_CONNECTION_CACHE_SIZE)
#define HPX_MAX_CONNECTION_CACHE_SIZE 64
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the LSB's of the global id's some of the components in a 
/// locality. The corresponding MSB's are set to the locality prefix
#define HPX_FACTORY_ID    1

#endif


