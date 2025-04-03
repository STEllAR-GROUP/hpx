This PR adds execution policy support to `run_on_all`, allowing for asynchronous execution and different execution policies.

Changes:
- Added new overloads for `run_on_all` that accept execution policies
- Support for all execution policies (par, seq, par_unseq, etc.)
- Return futures for asynchronous execution
- Added test cases to verify the new functionality
- Maintained backward compatibility with existing code

This implements the feature request from #6651, specifically adding support for:
```cpp
auto future = run_on_all(par(task), []() {...});
```

The changes are backward compatible and don't conflict with existing code. 