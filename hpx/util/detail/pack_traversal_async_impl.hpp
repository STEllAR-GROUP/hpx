//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_PACK_TRAVERSAL_ASYNC_IMPL_HPP
#define HPX_UTIL_DETAIL_PACK_TRAVERSAL_ASYNC_IMPL_HPP

#include <hpx/config.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/detail/container_category.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/intrusive_ptr.hpp>

#include <cstddef>
#include <exception>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx {
namespace util {
    namespace detail {
        /// A tag which is passed to the `operator()` of the visitor
        /// if an element is visited synchronously.
        struct async_traverse_visit_tag
        {
        };
        /// A tag which is passed to the `operator()` of the visitor
        /// if an element is visited after the traversal was detached.
        struct async_traverse_detach_tag
        {
        };
        /// A tag which is passed to the `operator()` of the visitor
        /// if the asynchronous pack traversal was finished.
        struct async_traverse_complete_tag
        {
        };

        /// A tag to identify that a mapper shall be constructed in-place
        /// from the first argument passed.
        template <typename T>
        struct async_traverse_in_place_tag
        {
        };

        /// Relocates the given pack with the given offset
        template <std::size_t Offset, typename Pack>
        struct relocate_index_pack;
        template <std::size_t Offset, std::size_t... Sequence>
        struct relocate_index_pack<Offset, pack_c<std::size_t, Sequence...>>
          : std::common_type<pack_c<std::size_t, (Sequence + Offset)...>>
        {
        };

        /// Creates a sequence from begin to end explicitly
        template <std::size_t Begin, std::size_t End>
        using explicit_range_sequence_of_t = typename relocate_index_pack<Begin,
            typename make_index_pack<End - Begin>::type>::type;

        /// Continues the traversal when the object is called
        template <typename Frame, typename State>
        class resume_traversal_callable
        {
            Frame frame_;
            State state_;

        public:
            explicit resume_traversal_callable(Frame frame, State state)
              : frame_(std::move(frame))
              , state_(std::move(state))
            {
            }

            /// The callable operator for resuming
            /// the asynchronous pack traversal
            void operator()();
        };

        /// Creates a resume_traversal_callable from the given frame and the
        /// given iterator tuple.
        template <typename Frame, typename State>
        auto make_resume_traversal_callable(Frame&& frame, State&& state)
            -> resume_traversal_callable<typename std::decay<Frame>::type,
                typename std::decay<State>::type>
        {
            return resume_traversal_callable<typename std::decay<Frame>::type,
                typename std::decay<State>::type>(
                std::forward<Frame>(frame), std::forward<State>(state));
        }

        /// Stores the visitor and the arguments to traverse
        template <typename Visitor, typename... Args>
        class async_traversal_frame : public Visitor
        {
            tuple<Args...> args_;

            Visitor& visitor() noexcept
            {
                return *static_cast<Visitor*>(this);
            }

            Visitor const& visitor() const noexcept
            {
                return *static_cast<Visitor const*>(this);
            }

        public:
            explicit async_traversal_frame(Visitor visitor, Args... args)
              : Visitor(std::move(visitor))
              , args_(util::make_tuple(std::move(args)...))
            {
            }

            template <typename MapperArg>
            explicit async_traversal_frame(async_traverse_in_place_tag<Visitor>,
                MapperArg&& mapper_arg, Args... args)
              : Visitor(std::forward<MapperArg>(mapper_arg))
              , args_(util::make_tuple(std::move(args)...))
            {
            }

            /// Returns the arguments of the frame
            tuple<Args...>& head() noexcept
            {
                return args_;
            }

            /// Calls the visitor with the given element
            template <typename T>
            auto traverse(T&& value) -> decltype(util::invoke(
                visitor(), async_traverse_visit_tag{}, std::forward<T>(value)))
            {
                return util::invoke(visitor(), async_traverse_visit_tag{},
                    std::forward<T>(value));
            }

            /// Calls the visitor with the given element and a continuation
            /// which is capable of continuing the asynchrone traversal
            /// when it's called later.
            template <typename T, typename Hierarchy>
            void async_continue(T&& value, Hierarchy&& hierarchy)
            {
                // Create a self reference
                boost::intrusive_ptr<async_traversal_frame> self(this);

                // Create a callable object which resumes the current
                // traversal when it's called.
                auto resumable = make_resume_traversal_callable(
                    std::move(self), std::forward<Hierarchy>(hierarchy));

                // Invoke the visitor with the current value and the
                // callable object to resume the control flow.
                util::invoke(visitor(), async_traverse_detach_tag{},
                    std::forward<T>(value), std::move(resumable));
            }

            /// Calls the visitor with no arguments to signalize that the
            /// asynchrnous traversal was finished.
            void async_complete()
            {
                util::invoke(
                    visitor(), async_traverse_complete_tag{}, std::move(args_));
            }
        };

        template <typename Target, std::size_t Begin, std::size_t End>
        struct static_async_range
        {
            Target* target_;

            HPX_CONSTEXPR auto operator*() const noexcept
                -> decltype(util::get<Begin>(*target_))
            {
                return util::get<Begin>(*target_);
            }

            template <std::size_t Position>
            HPX_CONSTEXPR static_async_range<Target, Position, End> relocate()
                const noexcept
            {
                return static_async_range<Target, Position, End>{target_};
            }

            HPX_CONSTEXPR static_async_range<Target, Begin + 1, End> next()
                const noexcept
            {
                return static_async_range<Target, Begin + 1, End>{target_};
            }

            HPX_CONSTEXPR bool is_finished() const noexcept
            {
                return false;
            }
        };
        /// Specialization for the end marker which doesn't provide
        /// a particular element dereference
        template <typename Target, std::size_t Begin>
        struct static_async_range<Target, Begin, Begin>
        {
            explicit static_async_range(Target*) {}

            HPX_CONSTEXPR bool is_finished() const noexcept
            {
                return true;
            }
        };

        /// Returns a static range for the given type
        template <typename T,
            typename Range = static_async_range<typename std::decay<T>::type,
                0U, util::tuple_size<typename std::decay<T>::type>::value>>
        Range make_static_range(T&& element)
        {
            auto pointer = std::addressof(element);
            return Range{pointer};
        }

        template <typename Begin, typename Sentinel>
        struct dynamic_async_range
        {
            Begin begin_;
            Sentinel sentinel_;

            dynamic_async_range& operator++() noexcept
            {
                ++begin_;
                return *this;
            }

            auto operator*() const noexcept
                -> decltype(*std::declval<Begin const&>())
            {
                return *begin_;
            }

            dynamic_async_range next() const
            {
                dynamic_async_range other = *this;
                ++other;
                return other;
            }

            bool is_finished() const
            {
                return begin_ == sentinel_;
            }
        };

        template <typename T>
        using dynamic_async_range_of_t = dynamic_async_range<
            typename std::decay<decltype(std::begin(std::declval<T>()))>::type,
            typename std::decay<decltype(std::end(std::declval<T>()))>::type>;

        /// Returns a dynamic range for the given type
        template <typename T, typename Range = dynamic_async_range_of_t<T>>
        Range make_dynamic_async_range(T&& element)
        {
            return Range{std::begin(element), std::end(element)};
        }

        /// Represents a particular point in a asynchronous traversal hierarchy
        template <typename Frame, typename... Hierarchy>
        class async_traversal_point
        {
            Frame frame_;
            tuple<Hierarchy...> hierarchy_;
            bool detached_;

        public:
            explicit async_traversal_point(
                Frame frame, tuple<Hierarchy...> hierarchy)
              : frame_(std::move(frame))
              , hierarchy_(std::move(hierarchy))
              , detached_(false)
            {
            }

            // Abort the current control flow
            void detach() noexcept
            {
                HPX_ASSERT(!detached_);
                detached_ = true;
            }

            /// Returns true when we should abort the current control flow
            bool is_detached() const noexcept
            {
                return detached_;
            }

            /// Creates a new traversal point which
            template <typename Parent>
            auto push(Parent&& parent) -> async_traversal_point<Frame,
                typename std::decay<Parent>::type, Hierarchy...>
            {
                // Create a new hierarchy which contains the
                // the parent (the last traversed element).
                auto hierarchy = util::tuple_cat(
                    util::make_tuple(std::forward<Parent>(parent)), hierarchy_);

                return async_traversal_point<Frame,
                    typename std::decay<Parent>::type, Hierarchy...>(
                    frame_, std::move(hierarchy));
            }

            /// Forks the current traversal point and continues the child
            /// of the given parent.
            template <typename Child, typename Parent>
            void fork(Child&& child, Parent&& parent)
            {
                // Push the parent on top of the hierarchy
                auto point = push(std::forward<Parent>(parent));

                // Continue the traversal with the current element
                point.async_traverse(std::forward<Child>(child));

                // Propagate the control detach back
                if (point.is_detached())
                {
                    detach();
                }
            }

            /// Async traverse a single element, and do nothing.
            /// This function is matched last.
            template <typename Matcher, typename Current>
            void async_traverse_one_impl(Matcher, Current&& current)
            {
                // Do nothing if the visitor doesn't accept the type
            }

            /// Async traverse a single element which isn't a container or
            /// tuple like type. This function is SFINAEd out if the element
            /// isn't accepted by the visitor.
            template <typename Current>
            auto async_traverse_one_impl(container_category_tag<false, false>,
                Current&& current)
                /// SFINAE this out, if the visitor doesn't accept
                /// the given element
                -> typename always_void<decltype(
                    std::declval<Frame>()->traverse(*current))>::type
            {
                if (!frame_->traverse(*current))
                {
                    // Store the current call hierarchy into a tuple for
                    // later reentrance.
                    auto state =
                        util::tuple_cat(util::make_tuple(current.next()),
                            std::move(hierarchy_));

                    // If the traversal method returns false, we detach the
                    // current execution context and call the visitor with the
                    // element and a continue callable object again.

                    frame_->async_continue(*current, std::move(state));

                    // Then detach the current execution context
                    detach();
                }
            }

            /// Async traverse a single element which is a container or
            /// tuple like type.
            template <bool IsTupleLike, typename Current>
            void async_traverse_one_impl(
                container_category_tag<true, IsTupleLike>,
                Current&& current)
            {
                fork(make_dynamic_async_range(*current),
                    std::forward<Current>(current));
            }

            /// Async traverse a single element which is a tuple like type only.
            template <typename Current>
            void async_traverse_one_impl(container_category_tag<false, true>,
                Current&& current)
            {
                fork(make_static_range(*current),
                    std::forward<Current>(current));
            }

            /// Async traverse the current iterator
            template <typename Current>
            void async_traverse_one(Current&& current)
            {
                using ElementType =
                    typename std::decay<decltype(*current)>::type;
                return async_traverse_one_impl(
                    container_category_of_t<ElementType>{},
                    std::forward<Current>(current));
            }

            /// Async traverse the current iterator but don't traverse
            /// if the control flow was detached.
            template <typename Current>
            void async_traverse_one_checked(Current&& current)
            {
                if (!is_detached())
                {
                    async_traverse_one(std::forward<Current>(current));
                }
            }

            template <std::size_t... Sequence, typename Current>
            void async_traverse_static_async_range(
                pack_c<std::size_t, Sequence...>,
                Current&& current)
            {
                int dummy[] = {0,
                    ((void) async_traverse_one_checked(
                         current.template relocate<Sequence>()),
                        0)...};
                (void) dummy;
            }

            /// Traverse a static range
            template <typename Target, std::size_t Begin, std::size_t End>
            void async_traverse(static_async_range<Target, Begin, End> current)
            {
                async_traverse_static_async_range(
                    explicit_range_sequence_of_t<Begin, End>{}, current);
            }

            /// Traverse a dynamic range
            template <typename Begin, typename Sentinel>
            void async_traverse(dynamic_async_range<Begin, Sentinel> range)
            {
                for (; !range.is_finished() && !is_detached(); ++range)
                {
                    async_traverse_one(range);
                }
            }
        };

        /// Deduces to the traversal point class of the
        /// given frame and hierarchy
        template <typename Frame, typename... Hierarchy>
        using traversal_point_of_t =
            async_traversal_point<typename std::decay<Frame>::type,
                typename std::decay<Hierarchy>::type...>;

        /// A callable object which is cabale of resuming an asynchronous
        /// pack traversal.
        struct resume_state_callable
        {
            /// Reenter an asynchronous iterator pack and continue
            /// its traversal.
            template <typename Frame, typename Current>
            void operator()(Frame&& frame, Current&& current) const
            {
                // Only process the next element if the current iterator
                // hasn't reached its end.
                if (!current.is_finished())
                {
                    traversal_point_of_t<Frame> point(
                        frame, util::make_tuple());

                    point.async_traverse(std::forward<Current>(current));

                    // Don't continue the frame when the execution was detached
                    if (point.is_detached())
                    {
                        return;
                    }
                }

                frame->async_complete();
            }

            /// Reenter an asynchronous iterator pack and continue
            /// its traversal.
            template <typename Frame, typename Current, typename Parent,
                typename... Hierarchy>
            void operator()(Frame&& frame, Current&& current, Parent&& parent,
                Hierarchy&&... hierarchy) const
            {
                // Only process the element if the current iterator
                // hasn't reached its end.
                if (!current.is_finished())
                {
                    // Don't forward the arguments here, since we still need
                    // the objects in a valid state later.
                    traversal_point_of_t<Frame, Parent, Hierarchy...> point(
                        frame, util::make_tuple(parent, hierarchy...));

                    point.async_traverse(std::forward<Current>(current));

                    // Don't continue the frame when the execution was detached
                    if (point.is_detached())
                    {
                        return;
                    }
                }

                // Pop the top element from the hierarchy, and shift the
                // parent element one to the right
                (*this)(std::forward<Frame>(frame),
                    std::forward<Parent>(parent).next(),
                    std::forward<Hierarchy>(hierarchy)...);
            }
        };

        template <typename Frame, typename State>
        void resume_traversal_callable<Frame, State>::operator()()
        {
            auto hierarchy =
                util::tuple_cat(util::make_tuple(frame_), std::move(state_));
            util::invoke_fused(resume_state_callable{}, std::move(hierarchy));
        }

        /// Gives access to types related to the traversal frame
        template <typename Visitor, typename... Args>
        struct async_traversal_types
        {
            /// Deduces to the async traversal frame type of the given
            /// traversal arguments and mapper
            using frame_type =
                async_traversal_frame<typename std::decay<Visitor>::type,
                    typename std::decay<Args>::type...>;

            /// The type of the frame pointer
            using frame_pointer_type = boost::intrusive_ptr<frame_type>;

            /// The type of the demoted visitor type
            using visitor_pointer_type = boost::intrusive_ptr<Visitor>;
        };
        template <typename Visitor, typename VisitorArg, typename... Args>
        struct async_traversal_types<async_traverse_in_place_tag<Visitor>,
            VisitorArg, Args...> : async_traversal_types<Visitor, Args...>
        {
        };

        /// Traverses the given pack with the given mapper
        template <typename Visitor, typename... Args,
            typename types = async_traversal_types<Visitor, Args...>>
        auto apply_pack_transform_async(Visitor&& visitor, Args&&... args) ->
            typename types::visitor_pointer_type
        {
            // Create the frame on the heap which stores the arguments
            // to traverse asynchronous.
            auto frame = [&] {
                auto ptr = new
                    typename types::frame_type(std::forward<Visitor>(visitor),
                        std::forward<Args>(args)...);

                // Create a intrusive_ptr from the heap object
                return typename types::frame_pointer_type(ptr);
            }();

            // Create a static range for the top level tuple
            auto range = make_static_range(frame->head());

            auto resumer = make_resume_traversal_callable(
                frame, util::make_tuple(std::move(range)));

            // Start the asynchronous traversal
            resumer();
            return frame;
        }
    }    // end namespace detail
}    // end namespace util
}    // end namespace hpx

#endif    // HPX_UTIL_DETAIL_PACK_TRAVERSAL_ASYNC_IMPL_HPP
