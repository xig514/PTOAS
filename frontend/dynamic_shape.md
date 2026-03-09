 Here is Claude's plan:
 Plan: Dynamic Shape Frontend with DynVar Support

 Context

 The current pto_frontend requires Tensor(dtype, ndim) which only specifies dimensionality, not shape. The user wants a new API similar to PyPTO's Tensor[[M, N], dtype] that:
 1. Specifies tensor shapes (static ints or dynamic DynVar variables)
 2. Allows dynamic variables to participate in arithmetic (loop bounds, offsets)
 3. Supports additional bool/float/int function parameters (already works)
 4. Tensor(dtype, ndim) API will be deprecated

 Dynamic scalar computations use the arith dialect (already used by ScalarValue), which naturally expresses integer/float arithmetic on SSA values.

 Files to Create

 1. frontend/pto_frontend/_dynvar.py — DynVar class

 class DynVar:
     """Dynamic shape variable for Tensor annotations."""

 - Holds a name and an optional _scalar: ScalarValue binding
 - During @kernel tracing, bound to the corresponding SSA index value
 - Supports arithmetic operators (__add__, __mul__, __floordiv__, __mod__, etc.) by delegating to the bound ScalarValue
 - Unbound after tracing completes (in a finally block)

 2. test/samples/frontend/DynamicAdd/test_dynamic_add.py — Test case

 Demonstrates:
 - DynVar for dynamic shapes
 - Tensor[[M, N], dtype] annotation
 - Dynamic int parameter
 - Nested for_range loops with dynamic bounds from DynVar arithmetic
 - Offset computation inside loops
 - Tile load/add/store with partitioned views
 - IR generation + ptoas compilation (--pto-level=level3)

 Files to Modify

 3. frontend/pto_frontend/_tensor.py

 - Add _TensorShapeSpec(shape, dtype) class (shape is list of int | DynVar)
 - Convert Tensor from function to class:
   - Tensor(dtype, ndim) → _TensorSpec (backward compat via __new__)
   - Tensor[[M, N], dtype] → _TensorShapeSpec (new, via __class_getitem__)

 4. frontend/pto_frontend/_kernel.py

 In KernelFunction._trace:
 - Handle _TensorShapeSpec alongside _TensorSpec when building flat_types (same flattening: ptr, index, index, ...)
 - When reconstructing proxies, bind each DynVar in the shape to its SSA value (first occurrence wins)
 - Unbind all DynVars in a finally block after tracing

 In KernelFunction._generate_caller_cpp:
 - Handle _TensorShapeSpec identically to _TensorSpec (same C++ signature)

 5. frontend/pto_frontend/_scalar.py

 In ScalarValue._coerce:
 - Add handling for DynVar → extract its bound SSA value

 6. frontend/pto_frontend/_utils.py

 In ensure_index_ssa:
 - Add handling for DynVar → delegate to its bound ScalarValue

 7. frontend/pto_frontend/__init__.py

 - Export DynVar from the package

 Key Design Decisions

 1. Same IR flattening: Tensor[[M, N], dtype] produces identical IR arguments to Tensor(dtype, 2) — each tensor still gets its own (ptr, dim0, dim1) args. DynVar sharing is a frontend convenience only.
 2. DynVar binding: First tensor using a DynVar binds it. This ensures the SSA value is available when the DynVar is used in arithmetic.
 3. arith dialect for dynamic computation: All dynamic scalar operations (loop bound calculation, offset computation) use arith.addi, arith.muli, arith.divsi, arith.remsi — already used by ScalarValue.
 4. Static sizes in partitions: x.partition(offsets=[dynamic], sizes=[64, 128]) produces !pto.partition_tensor_view<64x128xf16> with static sizes, which ptoas can parse.

 Verification

 # 1. Generate IR and verify it's valid
 cd test/samples/frontend/DynamicAdd
 python3 test_dynamic_add.py > dynamic_add.pto

 # 2. Compile IR with ptoas
 ptoas dynamic_add.pto --pto-level=level3

 The test script will also include inline assertions to verify IR correctness (contains scf.for, pto.partition_view, pto.tload, pto.tadd, etc.).

 Claude has written up a plan and is ready to execute. Would you like to proceed?

