//  Copyright (c) 2014 Anuj R. Sharma
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file src/components/vector/partition_vector_component.cpp

/// \brief Define the necessary component action boilerplate code.
///
/// This file defines the necessary component action boilerplate code for each
/// component action which is required for proper functioning of component
/// actions in the context of HPX.

#include <hpx/components/vector/partition_vector_component.hpp>

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::simple_component<
        hpx::server::partition_vector
    > partition_vector_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(partition_vector_type, partition_vector);

// Capacity related action registration
//
/** @brief Macro to define the boilerplate code for \a size component action.*/
HPX_REGISTER_ACTION(
    hpx::server::partition_vector::size_action,
    partition_vector_size_action);
/** @brief Macro to define the boilerplate code for \a max_size component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::partition_vector::max_size_action,
    partition_vector_max_size_action);
/** @brief Macro to define the boilerplate code for \a resize component action.
*/
HPX_REGISTER_ACTION(
    hpx::server::partition_vector::resize_action,
    partition_vector_resize_action);
/** @brief Macro to define the boilerplate code for \a capacity component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::partition_vector::capacity_action,
    partition_vector_capacity_action);
/** @brief Macro to define the boilerplate code for \a empty component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::partition_vector::empty_action,
    partition_vector_empty_action);
/** @brief Macro to define the boilerplate code for \a reserve component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::partition_vector::reserve_action,
    partition_vector_reserve_action);

// Element access component action registration

/** @brief Macro to define the boilerplate code for \a get_value component
 *          action.\
 */
HPX_REGISTER_ACTION(
    hpx::server::partition_vector::get_value_action,
    partition_vector_get_value_action);
/** @brief Macro to define the boilerplate code for \a front component action.*/
HPX_REGISTER_ACTION(
    hpx::server::partition_vector::front_action,
    partition_vector_front_action);
/** @brief Macro to define the boilerplate code for \a back component action.*/
HPX_REGISTER_ACTION(
    hpx::server::partition_vector::back_action,
    partition_vector_back_action);

// Modifiers component action registration

/** @brief Macro to define the boilerplate code for \a assign component action.
*/
HPX_REGISTER_ACTION(
    hpx::server::partition_vector::assign_action,
    partition_vector_assign_action);
/** @brief Macro to define the boilerplate code for \a push_back component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::partition_vector::push_back_action,
    partition_vector_push_back_action);
/** @brief Macro to define the boilerplate code for \a pop_back component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::partition_vector::pop_back_action,
    partition_vector_pop_back_action);
/** @brief Macro to define the boilerplate code for \a set_value component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::partition_vector::set_value_action,
    partition_vector_set_value_action);

/** @brief Macro to define the boilerplate code for \a clear component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::partition_vector::clear_action,
    partition_vector_clear_action);

// Algorithm API's component action registration

/** @brief Macro to define the boilerplate code for \a for_each component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::partition_vector::for_each_action,
    partition_vector_for_each_action);
