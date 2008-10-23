# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (hpxlib_HEADERS
    # global
    ../hpx/config.hpp
    ../hpx/exception.hpp
    ../hpx/exception_list.hpp
    ../hpx/hpx.hpp
    ../hpx/hpx_fwd.hpp
    ../hpx/version.hpp
    # config
    ../hpx/config/defaults.hpp
    ../hpx/config/export_definitions.hpp
    ../hpx/config/warnings_prefix.hpp
    ../hpx/config/warnings_suffix.hpp
    # include
    ../hpx/include/actions.hpp
    ../hpx/include/applier.hpp
    ../hpx/include/components.hpp
    ../hpx/include/lcos.hpp
    ../hpx/include/naming.hpp
    ../hpx/include/parcelset.hpp
    ../hpx/include/runtime.hpp
    ../hpx/include/threadmanager.hpp
    ../hpx/include/util.hpp
    # LCO's
    ../hpx/lcos/barrier.hpp
    ../hpx/lcos/base_lco.hpp
    ../hpx/lcos/condition.hpp
    ../hpx/lcos/counting_semaphore.hpp
    ../hpx/lcos/eager_future.hpp
    ../hpx/lcos/eager_future_constructors.hpp
    ../hpx/lcos/eager_future_constructors_direct.hpp
    ../hpx/lcos/future_value.hpp
    ../hpx/lcos/lazy_future.hpp
    ../hpx/lcos/lazy_future_get_results.hpp
    ../hpx/lcos/lazy_future_get_results_direct.hpp
    ../hpx/lcos/mutex.hpp
    # runtime
    ../hpx/runtime/get_lva.hpp
    ../hpx/runtime/runtime.hpp
    # actions 
    ../hpx/runtime/actions/action.hpp
    ../hpx/runtime/actions/action_constructors.hpp
    ../hpx/runtime/actions/action_implementations.hpp
    ../hpx/runtime/actions/continuation.hpp
    ../hpx/runtime/actions/continuation_impl.hpp
    ../hpx/runtime/actions/action_manager.hpp
    # applier
    ../hpx/runtime/applier/applier.hpp
    ../hpx/runtime/applier/applier_implementations.hpp
    ../hpx/runtime/applier/apply_helper.hpp
    ../hpx/runtime/applier/apply_helper_implementations.hpp
    # components
    ../hpx/runtime/components/component_type.hpp
    ../hpx/runtime/components/component_factory_base.hpp
    ../hpx/runtime/components/component_factory.hpp
    ../hpx/runtime/components/generic_component.hpp
    ../hpx/runtime/components/generic_component_eval.hpp
    ../hpx/runtime/components/memory_block.hpp
    ../hpx/runtime/components/runtime_support.hpp
    ../hpx/runtime/components/server/generic_component.hpp
    ../hpx/runtime/components/server/generic_component_implementation.hpp
    ../hpx/runtime/components/server/manage_component.hpp
    ../hpx/runtime/components/server/managed_component_base.hpp
    ../hpx/runtime/components/server/memory.hpp
    ../hpx/runtime/components/server/memory_block.hpp
    ../hpx/runtime/components/server/runtime_support.hpp
    ../hpx/runtime/components/server/simple_component_base.hpp
    ../hpx/runtime/components/server/wrapper_heap.hpp
    ../hpx/runtime/components/server/wrapper_heap_list.hpp
    ../hpx/runtime/components/stubs/generic_component.hpp
    ../hpx/runtime/components/stubs/generic_component_eval.hpp
    ../hpx/runtime/components/stubs/memory_block.hpp
    ../hpx/runtime/components/stubs/runtime_support.hpp
    # naming
    ../hpx/runtime/naming/address.hpp
    ../hpx/runtime/naming/locality.hpp
    ../hpx/runtime/naming/name.hpp
    ../hpx/runtime/naming/resolver_client.hpp
    ../hpx/runtime/naming/resolver_client_connection.hpp
    ../hpx/runtime/naming/resolver_server.hpp
    ../hpx/runtime/naming/server/connection.hpp
    ../hpx/runtime/naming/server/reply.hpp
    ../hpx/runtime/naming/server/request.hpp
    ../hpx/runtime/naming/server/request_handler.hpp
    # parcelset
    ../hpx/runtime/parcelset/connection_cache.hpp
    ../hpx/runtime/parcelset/parcel.hpp
    ../hpx/runtime/parcelset/parcelhandler.hpp
    ../hpx/runtime/parcelset/parcelport.hpp
    ../hpx/runtime/parcelset/parcelport_connection.hpp
    ../hpx/runtime/parcelset/server/parcelhandler_queue.hpp
    ../hpx/runtime/parcelset/server/parcelport_queue.hpp
    ../hpx/runtime/parcelset/server/parcelport_server_connection.hpp
    # threadmanager
    ../hpx/runtime/threads/thread.hpp
    ../hpx/runtime/threads/threadmanager.hpp
    # util
    ../hpx/util/asio_util.hpp
    ../hpx/util/binary_portable_iarchive.hpp
    ../hpx/util/binary_portable_oarchive.hpp
    ../hpx/util/container_device.hpp
    ../hpx/util/find_msb.hpp
    ../hpx/util/full_empty_memory.hpp
    ../hpx/util/full_empty_store.hpp
    ../hpx/util/future.hpp
    ../hpx/util/generate_unique_ids.hpp
    ../hpx/util/high_resolution_timer.hpp
    ../hpx/util/ini.hpp
    ../hpx/util/init_ini_data.hpp
    ../hpx/util/io_service_pool.hpp
    ../hpx/util/logging.hpp
    ../hpx/util/one_size_heap_list.hpp
    ../hpx/util/portable_binary_archive.hpp
    ../hpx/util/portable_binary_iarchive.hpp
    ../hpx/util/portable_binary_oarchive.hpp
    ../hpx/util/runtime_configuration.hpp
    ../hpx/util/safe_bool.hpp
    ../hpx/util/serialize_sequence.hpp
    ../hpx/util/static.hpp
    ../hpx/util/unlock_lock.hpp
    ../hpx/util/util.hpp)

set (hpxlib_SOURCES
    ../src/exception_list.cpp
    # lcos
    ../src/lcos/base_lco.cpp
    # runtime
    ../src/runtime/runtime.cpp
    # action manager
    ../src/runtime/actions/action_manager.cpp
    ../src/runtime/actions/continuation.cpp
    # applier
    ../src/runtime/applier/applier.cpp
    # components
    ../src/runtime/components/component_type.cpp
    ../src/runtime/components/server/memory.cpp
    ../src/runtime/components/server/memory_block.cpp
    ../src/runtime/components/server/runtime_support.cpp
    # naming 
    ../src/runtime/naming/address.cpp
    ../src/runtime/naming/locality.cpp
    ../src/runtime/naming/name.cpp
    ../src/runtime/naming/resolver_client.cpp
    ../src/runtime/naming/resolver_server.cpp
    ../src/runtime/naming/server/reply.cpp
    ../src/runtime/naming/server/request.cpp
    ../src/runtime/naming/server/request_handler.cpp
    # parcelset
    ../src/runtime/parcelset/connection_cache.cpp
    ../src/runtime/parcelset/parcel.cpp
    ../src/runtime/parcelset/parcelhandler.cpp
    ../src/runtime/parcelset/parcelport.cpp
    ../src/runtime/parcelset/server/parcelhandler_queue.cpp
    # threadmanager
    ../src/runtime/threads/thread.cpp
    ../src/runtime/threads/threadmanager.cpp
    # util 
    ../src/util/asio_util.cpp
    ../src/util/ini.cpp
    ../src/util/io_service_pool.cpp
    ../src/util/logging.cpp
    ../src/util/runtime_configuration.cpp)

# add source file from coroutine library (needed for gcc only)
if (CMAKE_COMPILER_IS_GNUCXX)
    SET(hpxlib_SOURCES 
        ${hpxlib_SOURCES} 
        ../external/coroutine/libs/coroutine/src/swapcontext.cpp)
endif()

# separate shared module containing our portable serialization archives
set (hpxserializationlib_HEADERS
    ../hpx/hpx_fwd.hpp
    ../hpx/util/binary_portable_iarchive.hpp
    ../hpx/util/binary_portable_oarchive.hpp
    ../hpx/util/portable_binary_archive.hpp
    ../hpx/util/portable_binary_iarchive.hpp
    ../hpx/util/portable_binary_oarchive.hpp)

set (hpxserializationlib_SOURCES
    util/binary_portable_archives.cpp
    util/portable_binary_iarchive.cpp
    util/portable_binary_oarchive.cpp)

add_definitions(-DHPX_EXPORTS)

# build instructions
add_library(hpx_serialization SHARED 
    ${hpxserializationlib_SOURCES} ${hpxserializationlib_HEADERS})
target_link_libraries(hpx_serialization ${Boost_LIBRARIES})

add_library(hpx SHARED 
    ${hpxlib_SOURCES} ${hpxlib_HEADERS})
target_link_libraries(hpx hpx_serialization ${Boost_LIBRARIES})

# installation instructions
install(TARGETS hpx hpx_serialization
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)


#set_target_properties(lib_${PROJECT_NAME} PROPERTIES
# create *nix style library versions + symbolic links
#   VERSION ${${PROJECT_NAME}_VERSION}
#   SOVERSION ${${PROJECT_NAME}_SOVERSION}
# allow creating static and shared libs without conflicts
#   CLEAN_DIRECT_OUTPUT 1
# avoid conflicts between library and binary target names
#   OUTPUT_NAME ${PROJECT_NAME})
