//  Copyright (c) 2014 Anuj R. Sharma
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/** @file src/components/vector/chunk_vector_component.cpp
 *
 *  @brief Define the necessary component action boilerplate code.
 *
 *  This file define the necessary component action boilerplate code for each
 *   component action which is required for proper functioning of component
 *   actions in the context of HPX.
 */
#include <hpx/components/vector/chunk_vector_component.hpp>
HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<hpx::server::chunk_vector>
                                                    chunk_vector_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(chunk_vector_type, chunk_vector);

//Capacity related action registration
/** @brief Macro to define the boilerplate code for \a size component action.*/
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::size_action,
    chunk_vector_size_action);
/** @brief Macro to define the boilerplate code for \a max_size component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::max_size_action,
    chunk_vector_max_size_action);
/** @brief Macro to define the boilerplate code for \a resize component action.
*/
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::resize_action,
    chunk_vector_resize_action);
/** @brief Macro to define the boilerplate code for \a capacity component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::capacity_action,
    chunk_vector_capacity_action);
/** @brief Macro to define the boilerplate code for \a empty component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::empty_action,
    chunk_vector_empty_action);
/** @brief Macro to define the boilerplate code for \a reserve component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::reserve_action,
    chunk_vector_reserve_action);

//Element access component action registration
/** @brief Macro to define the boilerplate code for \a get_value_noexpt
 *          component action.
 */
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::get_value_noexpt_action,
    chunk_vector_get_value_noexpt_action);
/** @brief Macro to define the boilerplate code for \a get_value component
 *          action.\
 */
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::get_value_action,
    chunk_vector_get_value_action);
/** @brief Macro to define the boilerplate code for \a front component action.*/
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::front_action,
    chunk_vector_front_action);
/** @brief Macro to define the boilerplate code for \a back component action.*/
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::back_action,
    chunk_vector_back_action);

////Modifiers component action registration
/** @brief Macro to define the boilerplate code for \a assign component action.
*/
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::assign_action,
    chunk_vector_assign_action);
/** @brief Macro to define the boilerplate code for \a push_back component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::push_back_action,
    chunk_vector_push_back_action);
/** @brief Macro to define the boilerplate code for \a push_back_rval component
*           action.
*/
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::push_back_rval_action,
    chunk_vector_push_back_rval_action);
/** @brief Macro to define the boilerplate code for \a pop_back component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::pop_back_action,
    chunk_vector_pop_back_action);
/** @brief Macro to define the boilerplate code for \a set_value component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::set_value_action,
    chunk_vector_set_value_action);
/** @brief Macro to define the boilerplate code for \a set_value_rval component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::set_value_rval_action,
    chunk_vector_set_value_rval_action);

/** @brief Macro to define the boilerplate code for \a clear component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::clear_action,
    chunk_vector_clear_action);

//Algorithm API's component action registration
/** @brief Macro to define the boilerplate code for \a chunk_for_each component
 *          action.
 */
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::chunk_for_each_action,
    chunk_vector_chunk_for_each_action);
/** @brief Macro to define the boilerplate code for \a chunk_for_each_const
 *          component action.
 */
HPX_REGISTER_ACTION(
    hpx::server::chunk_vector::chunk_for_each_const_action,
    chunk_vector_chunk_for_each_const_action);

