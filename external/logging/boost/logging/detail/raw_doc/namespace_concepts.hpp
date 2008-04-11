namespace boost { namespace logging {

/** 
@page namespace_concepts Concepts as namespaces

- @ref namespace_general
- @ref namespace_process 
- @ref namespace_manipulator
- @ref namespace_write

This library uses a few concepts. 

Take filter, for instance. It's a very simple concept - it tells you if "it's enabled".

Each concept can be implemented in several ways. To make it easier for you, <b>each concept is a namespace</b>.
In the given namespace, you'll find possible implementations of that concept. Of course, to those implementations, you can add your own ;)

\n\n
@section namespace_general General concepts
- filter - available filter implementations
- level - in case you want to use Log Levels
- writer - %writer objects; they do the actual write of the message 
- scenario - in case you want to easily specify the logger and filter class(es), based on your application's needs

\n\n
@section namespace_process Logging (Processing) the message 
(for more info, see logger class)
- gather - gathering the message
- writer - %writer objects; they do the actual write of the message 
- gather::ostream_like - (related to gathering the message) allows gathering the message using the cool operator<< (@ref workflow_2a)


\n\n
@section namespace_manipulator Manipulator concepts
- manipulator - what a manipulator is: a formatter or a destination
- formatter - available formatters
- destination - available destinations
- tag - available tags
- formatter::tag - available tag formatters

\n\n
@section namespace_write Writing concepts
- format_and_write - contains the logic for formatting and writing to destinations
- msg_route - contains the logic for routing the message to the formatters and destinations
- op_equal - implements operator==, in order to compare formatters and/or destinations. Useful when you want to 
             erase formatters/destinations from a logger.
- optimize - (related to gathering the message) optimizes holding the message, as it's formatted. Formatting can modify the message.
                     Implementations from this namespace allow optimizing the medium so that modifying the message is as fast as possible



*/

}}
