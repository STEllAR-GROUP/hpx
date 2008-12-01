# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (hpxlib_HEADERS
    # global
    ${hpx_SOURCE_DIR}/hpx/config.hpp
    ${hpx_SOURCE_DIR}/hpx/exception.hpp
    ${hpx_SOURCE_DIR}/hpx/exception_list.hpp
    ${hpx_SOURCE_DIR}/hpx/hpx.hpp
    ${hpx_SOURCE_DIR}/hpx/hpx_fwd.hpp
    ${hpx_SOURCE_DIR}/hpx/version.hpp
    # config
    ${hpx_SOURCE_DIR}/hpx/config/defaults.hpp
    ${hpx_SOURCE_DIR}/hpx/config/export_definitions.hpp
    ${hpx_SOURCE_DIR}/hpx/config/warnings_prefix.hpp
    ${hpx_SOURCE_DIR}/hpx/config/warnings_suffix.hpp
    # include
    ${hpx_SOURCE_DIR}/hpx/include/actions.hpp
    ${hpx_SOURCE_DIR}/hpx/include/applier.hpp
    ${hpx_SOURCE_DIR}/hpx/include/components.hpp
    ${hpx_SOURCE_DIR}/hpx/include/lcos.hpp
    ${hpx_SOURCE_DIR}/hpx/include/naming.hpp
    ${hpx_SOURCE_DIR}/hpx/include/parcelset.hpp
    ${hpx_SOURCE_DIR}/hpx/include/runtime.hpp
    ${hpx_SOURCE_DIR}/hpx/include/threadmanager.hpp
    ${hpx_SOURCE_DIR}/hpx/include/util.hpp
    # LCO's
    ${hpx_SOURCE_DIR}/hpx/lcos/barrier.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/base_lco.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/condition.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/counting_semaphore.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/eager_future.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/eager_future_constructors.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/eager_future_constructors_direct.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/future_value.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/future_wait.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/lazy_future.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/lazy_future_get_results.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/lazy_future_get_results_direct.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/mutex.hpp
    # runtime
    ${hpx_SOURCE_DIR}/hpx/runtime/get_lva.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/runtime.hpp
    # actions 
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/action_constructors.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/action_manager.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/action_support.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/component_action.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/component_action_implementations.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/continuation.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/continuation_impl.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/plain_action.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/plain_action_implementations.hpp
    # applier
    ${hpx_SOURCE_DIR}/hpx/runtime/applier/applier.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/applier/apply_helper.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/applier/apply_helper_implementations.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/applier/apply.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/applier/apply_implementations.hpp
    # components
    ${hpx_SOURCE_DIR}/hpx/runtime/components/client_base.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/component_type.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/component_factory_base.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/component_factory.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/derived_component_factory.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/memory_block.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/runtime_support.hpp

    ${hpx_SOURCE_DIR}/hpx/runtime/components/server/manage_component.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/server/managed_component_base.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/server/memory.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/server/memory_block.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/server/plain_function.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/server/runtime_support.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/server/simple_component_base.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/server/wrapper_heap.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/server/wrapper_heap_list.hpp

    ${hpx_SOURCE_DIR}/hpx/runtime/components/stubs/memory_block.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/stubs/runtime_support.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/stubs/stub_base.hpp
    # naming
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/address.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/locality.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/name.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/resolver_client.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/resolver_client_connection.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/resolver_server.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/server/connection.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/server/reply.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/server/request.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/server/request_handler.hpp
    # parcelset
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/parcel.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/parcelhandler.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/parcelport.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/parcelport_connection.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/server/parcelhandler_queue.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/server/parcelport_queue.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/server/parcelport_server_connection.hpp
    # threadmanager
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/thread.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/thread_affinity.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/thread_helpers.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/threadmanager.hpp
    # util
    ${hpx_SOURCE_DIR}/hpx/util/asio_util.hpp
    ${hpx_SOURCE_DIR}/hpx/util/binary_portable_iarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/binary_portable_oarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/block_profiler.hpp
    ${hpx_SOURCE_DIR}/hpx/util/connection_cache.hpp
    ${hpx_SOURCE_DIR}/hpx/util/container_device.hpp
    ${hpx_SOURCE_DIR}/hpx/util/find_msb.hpp
    ${hpx_SOURCE_DIR}/hpx/util/full_empty_memory.hpp
    ${hpx_SOURCE_DIR}/hpx/util/full_empty_store.hpp
    ${hpx_SOURCE_DIR}/hpx/util/future.hpp
    ${hpx_SOURCE_DIR}/hpx/util/generate_unique_ids.hpp
    ${hpx_SOURCE_DIR}/hpx/util/high_resolution_timer.hpp
    ${hpx_SOURCE_DIR}/hpx/util/ini.hpp
    ${hpx_SOURCE_DIR}/hpx/util/init_ini_data.hpp
    ${hpx_SOURCE_DIR}/hpx/util/io_service_pool.hpp
    ${hpx_SOURCE_DIR}/hpx/util/logging.hpp
    ${hpx_SOURCE_DIR}/hpx/util/one_size_heap_list.hpp
    ${hpx_SOURCE_DIR}/hpx/util/portable_binary_archive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/portable_binary_iarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/portable_binary_oarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/runtime_configuration.hpp
    ${hpx_SOURCE_DIR}/hpx/util/safe_bool.hpp
    ${hpx_SOURCE_DIR}/hpx/util/serialize_sequence.hpp
    ${hpx_SOURCE_DIR}/hpx/util/static.hpp
    ${hpx_SOURCE_DIR}/hpx/util/time_logger.hpp
    ${hpx_SOURCE_DIR}/hpx/util/unlock_lock.hpp
    ${hpx_SOURCE_DIR}/hpx/util/util.hpp)

set (hpxlib_SOURCES
    ${hpx_SOURCE_DIR}/src/exception_list.cpp
    # lcos
    ${hpx_SOURCE_DIR}/src/lcos/base_lco.cpp
    # runtime
    ${hpx_SOURCE_DIR}/src/runtime/runtime.cpp
    # action manager
    ${hpx_SOURCE_DIR}/src/runtime/actions/action_manager.cpp
    ${hpx_SOURCE_DIR}/src/runtime/actions/continuation.cpp
    # applier
    ${hpx_SOURCE_DIR}/src/runtime/applier/applier.cpp
    # components
    ${hpx_SOURCE_DIR}/src/runtime/components/component_type.cpp
    ${hpx_SOURCE_DIR}/src/runtime/components/server/memory.cpp
    ${hpx_SOURCE_DIR}/src/runtime/components/server/memory_block.cpp
    ${hpx_SOURCE_DIR}/src/runtime/components/server/plain_function.cpp
    ${hpx_SOURCE_DIR}/src/runtime/components/server/runtime_support.cpp
    # naming 
    ${hpx_SOURCE_DIR}/src/runtime/naming/address.cpp
    ${hpx_SOURCE_DIR}/src/runtime/naming/locality.cpp
    ${hpx_SOURCE_DIR}/src/runtime/naming/name.cpp
    ${hpx_SOURCE_DIR}/src/runtime/naming/resolver_client.cpp
    ${hpx_SOURCE_DIR}/src/runtime/naming/resolver_server.cpp
    ${hpx_SOURCE_DIR}/src/runtime/naming/server/reply.cpp
    ${hpx_SOURCE_DIR}/src/runtime/naming/server/request.cpp
    ${hpx_SOURCE_DIR}/src/runtime/naming/server/request_handler.cpp
    # parcelset
    ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcel.cpp
    ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcelhandler.cpp
    ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcelport.cpp
    ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcelport_connection.cpp
    ${hpx_SOURCE_DIR}/src/runtime/parcelset/server/parcelhandler_queue.cpp
    # threadmanager
    ${hpx_SOURCE_DIR}/src/runtime/threads/thread.cpp
    ${hpx_SOURCE_DIR}/src/runtime/threads/threadmanager.cpp
    # util 
    ${hpx_SOURCE_DIR}/src/util/asio_util.cpp
    ${hpx_SOURCE_DIR}/src/util/ini.cpp
    ${hpx_SOURCE_DIR}/src/util/io_service_pool.cpp
    ${hpx_SOURCE_DIR}/src/util/logging.cpp
    ${hpx_SOURCE_DIR}/src/util/runtime_configuration.cpp)

# add source file from coroutine library (needed for gcc only)
if (CMAKE_COMPILER_IS_GNUCXX)
    SET(hpxlib_SOURCES 
        ${hpxlib_SOURCES} 
        ${hpx_SOURCE_DIR}/external/coroutine/libs/coroutine/src/swapcontext.cpp)
endif()

# yes, the main shared library contains components as well
add_definitions(-DHPX_COMPONENT_NAME=hpx)
add_definitions(-DHPX_EXPORTS)
add_definitions(-DBOOST_COROUTINE_EXPORTS)

# separate shared module containing our portable serialization archives
set (hpxserializationlib_HEADERS
    ${hpx_SOURCE_DIR}/hpx/hpx_fwd.hpp
    ${hpx_SOURCE_DIR}/hpx/util/binary_portable_iarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/binary_portable_oarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/portable_binary_archive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/portable_binary_iarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/portable_binary_oarchive.hpp)

set (hpxserializationlib_SOURCES
    util/binary_portable_archives.cpp
    util/portable_binary_iarchive.cpp
    util/portable_binary_oarchive.cpp)

# build instructions for serialization support library
add_library(hpx_serialization SHARED 
    ${hpxserializationlib_SOURCES} ${hpxserializationlib_HEADERS})
set_target_properties(hpx_serialization PROPERTIES
    VERSION ${HPX_VERSION}      # create *nix style library versions + symbolic links
    SOVERSION ${HPX_SOVERSION}
    CLEAN_DIRECT_OUTPUT 1       # allow creating static and shared libs without conflicts
    OUTPUT_NAME ${component_LIBRARY_PREFIX}hpx_serialization)
target_link_libraries(hpx_serialization ${Boost_LIBRARIES})

# build instructions for main shared HPX library
add_library(hpx SHARED 
    ${hpxlib_SOURCES} ${hpxlib_HEADERS})
set_target_properties(hpx PROPERTIES
    VERSION ${HPX_VERSION}      # create *nix style library versions + symbolic links
    SOVERSION ${HPX_SOVERSION}
    CLEAN_DIRECT_OUTPUT 1       # allow creating static and shared libs without conflicts
    OUTPUT_NAME ${component_LIBRARY_PREFIX}hpx)
target_link_libraries(hpx hpx_serialization ${Boost_LIBRARIES})

# installation instructions
install(TARGETS hpx hpx_serialization
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)


