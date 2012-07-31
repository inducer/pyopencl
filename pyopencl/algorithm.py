"""Scan primitive."""

from __future__ import division

__copyright__ = """
Copyright 2011-2012 Andreas Kloeckner

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.scan import GenericScanKernel
from pyopencl.tools import dtype_to_ctype
from pyopencl.tools import context_dependent_memoize
from pytools import memoize
from mako.template import Template




# {{{ copy_if

@context_dependent_memoize
def _get_copy_if_kernel(ctx, dtype, predicate, scan_dtype,
        extra_args_types, preamble):
    ctype = dtype_to_ctype(dtype)
    arguments = [
        "__global %s *ary" % ctype,
        "__global %s *out" % ctype,
        "__global unsigned long *count",
        ] + [
                "%s %s" % (dtype_to_ctype(arg_dtype), name)
                for name, arg_dtype in extra_args_types]

    return GenericScanKernel(
            ctx, dtype,
            arguments=", ".join(arguments),
            input_expr="(%s) ? 1 : 0" % predicate,
            scan_expr="a+b", neutral="0",
            output_statement="""
                if (prev_item != item) out[item-1] = ary[i];
                if (i+1 == N) *count = item;
                """,
            preamble=preamble)

def copy_if(ary, predicate, extra_args=[], queue=None, preamble=""):
    """Copy the elements of *ary* satisfying *predicate* to an output array.

    :arg predicate: a C expression evaluating to a `bool`, represented as a string.
        The value to test is available as `ary[i]`, and if the expression evaluates
        to `true`, then this value ends up in the output.
    :arg extra_args: |scan_extra_args|
    :arg preamble: |preamble|
    :returns: a tuple *(out, count)* where *out* is the output array and *count*
        is an on-device scalar (fetch to host with `count.get()`) indicating
        how many elements satisfied *predicate*.
    """
    if len(ary) > np.iinfo(np.uint32).max:
        scan_dtype = np.uint64
    else:
        scan_dtype = np.uint32

    extra_args_types = tuple((name, val.dtype) for name, val in extra_args)
    extra_args_values = tuple(val for name, val in extra_args)

    knl = _get_copy_if_kernel(ary.context, ary.dtype, predicate, scan_dtype,
            extra_args_types, preamble=preamble)
    out = cl.array.empty_like(ary)
    count = ary._new_with_changes(data=None, shape=(), strides=(), dtype=np.uint64)
    knl(ary, out, count, *extra_args_values, queue=queue)
    return out, count

# }}}

# {{{ remove_if

def remove_if(ary, predicate, extra_args=[], queue=None, preamble=""):
    """Copy the elements of *ary* not satisfying *predicate* to an output array.

    :arg predicate: a C expression evaluating to a `bool`, represented as a string.
        The value to test is available as `ary[i]`, and if the expression evaluates
        to `false`, then this value ends up in the output.
    :arg extra_args: |scan_extra_args|
    :arg preamble: |preamble|
    :returns: a tuple *(out, count)* where *out* is the output array and *count*
        is an on-device scalar (fetch to host with `count.get()`) indicating
        how many elements did not satisfy *predicate*.
    """
    return copy_if(ary, "!(%s)" % predicate, extra_args=extra_args, queue=queue,
            preamble=preamble)

# }}}

# {{{ partition

@context_dependent_memoize
def _get_partition_kernel(ctx, dtype, predicate, scan_dtype,
        extra_args_types, preamble):
    ctype = dtype_to_ctype(dtype)
    arguments = [
        "__global %s *ary" % ctype,
        "__global %s *out_true" % ctype,
        "__global %s *out_false" % ctype,
        "__global unsigned long *count_true",
        ] + [
                "%s %s" % (dtype_to_ctype(arg_dtype), name)
                for name, arg_dtype in extra_args_types]

    return GenericScanKernel(
            ctx, dtype,
            arguments=", ".join(arguments),
            input_expr="(%s) ? 1 : 0" % predicate,
            scan_expr="a+b", neutral="0",
            output_statement="""
                if (prev_item != item)
                    out_true[item-1] = ary[i];
                else
                    out_false[i-item] = ary[i];
                if (i+1 == N) *count_true = item;
                """,
            preamble=preamble)

def partition(ary, predicate, extra_args=[], queue=None, preamble=""):
    """Copy the elements of *ary* into one of two arrays depending on whether
    they satisfy *predicate*.

    :arg predicate: a C expression evaluating to a `bool`, represented as a string.
        The value to test is available as `ary[i]`.
    :arg extra_args: |scan_extra_args|
    :arg preamble: |preamble|
    :returns: a tuple *(out_true, out_false, count)* where *count*
        is an on-device scalar (fetch to host with `count.get()`) indicating
        how many elements satisfied the predicate.
    """
    if len(ary) > np.iinfo(np.uint32).max:
        scan_dtype = np.uint64
    else:
        scan_dtype = np.uint32

    extra_args_types = tuple((name, val.dtype) for name, val in extra_args)
    extra_args_values = tuple(val for name, val in extra_args)

    knl = _get_partition_kernel(ary.context, ary.dtype, predicate, scan_dtype,
            extra_args_types, preamble)
    out_true = cl.array.empty_like(ary)
    out_false = cl.array.empty_like(ary)
    count = ary._new_with_changes(data=None, shape=(), strides=(), dtype=np.uint64)
    knl(ary, out_true, out_false, count, *extra_args_values, queue=queue)
    return out_true, out_false, count

# }}}

# {{{ unique

@context_dependent_memoize
def _get_unique_kernel(ctx, dtype, is_equal_expr, scan_dtype,
        extra_args_types, preamble):
    ctype = dtype_to_ctype(dtype)
    arguments = [
        "__global %s *ary" % ctype,
        "__global %s *out" % ctype,
        "__global unsigned long *count_unique",
        ] + [
                "%s %s" % (dtype_to_ctype(arg_dtype), name)
                for name, arg_dtype in extra_args_types]

    from pyopencl.scan import _process_code_for_macro
    key_expr_define = "#define IS_EQUAL_EXPR(a, b) %s\n" \
            % _process_code_for_macro(is_equal_expr)
    return GenericScanKernel(
            ctx, dtype,
            arguments=", ".join(arguments),
            input_fetch_exprs=[
                ("ary_im1", "ary", -1),
                ("ary_i", "ary", 0),
                ],
            input_expr="(i == 0) || (IS_EQUAL_EXPR(ary_im1, ary_i) ? 0 : 1)",
            scan_expr="a+b", neutral="0",
            output_statement="""
                if (prev_item != item) out[item-1] = ary[i];
                if (i+1 == N) *count_unique = item;
                """,
            preamble=preamble+"\n\n"+key_expr_define)

def unique(ary, is_equal_expr="a == b", extra_args=[], queue=None, preamble=""):
    """Copy the elements of *ary* into the output if *is_equal_expr*, applied to the
    array element and its predecessor, yields false.

    Works like the UNIX command :program:`uniq`, with a potentially custom comparison.
    This operation is often used on sorted sequences.

    :arg is_equal_expr: a C expression evaluating to a `bool`, represented as a string.
        The elements being compared are available as `a` and `b`. If this expression
        yields `false`, the two are considered distinct.
    :arg extra_args: |scan_extra_args|
    :arg preamble: |preamble|
    :returns: a tuple *(out, count)* where *out* is the output array and *count*
        is an on-device scalar (fetch to host with `count.get()`) indicating
        how many elements satisfied the predicate.
    """

    if len(ary) > np.iinfo(np.uint32).max:
        scan_dtype = np.uint64
    else:
        scan_dtype = np.uint32

    extra_args_types = tuple((name, val.dtype) for name, val in extra_args)
    extra_args_values = tuple(val for name, val in extra_args)

    knl = _get_unique_kernel(ary.context, ary.dtype, is_equal_expr, scan_dtype,
            extra_args_types, preamble)
    out = cl.array.empty_like(ary)
    count = ary._new_with_changes(data=None, shape=(), strides=(), dtype=np.uint64)
    knl(ary, out, count, *extra_args_values, queue=queue)
    return out, count

# }}}

# {{{ radix_sort

def _padded_bin(i, l):
    s = bin(i)[2:]
    while len(s) < l:
        s = '0' + s
    return s

@memoize
def _make_sort_scan_type(device, bits, index_dtype):
    fields = []
    for mnr in range(2**bits):
        fields.append(('c%s' % _padded_bin(mnr, bits), index_dtype))

    dtype = np.dtype(fields)

    name = "pyopencl_sort_scan_%s_%dbits_t" % (
            index_dtype.type.__name__, bits)
    from pyopencl.tools import register_dtype, match_dtype_to_c_struct
    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)

    register_dtype(dtype, name)
    return name, dtype, c_decl

# {{{ types, helpers preamble

RADIX_SORT_PREAMBLE_TPL = Template(r"""//CL//
    typedef ${scan_ctype} scan_t;
    typedef ${key_ctype} key_t;
    typedef ${index_ctype} index_t;

    // #define DEBUG
    #ifdef DEBUG
        #define dbg_printf(ARGS) printf ARGS
    #else
        #define dbg_printf(ARGS) /* */
    #endif

    <%
      def get_count_branch(known_bits):
          if len(known_bits) == bits:
              return "s.c%s" % known_bits

          b = len(known_bits)
          boundary_mnr = known_bits + "1" + (bits-b-1)*"0"

          return ("((mnr < %s) ? %s : %s)" % (
              int(boundary_mnr, 2),
              get_count_branch(known_bits+"0"),
              get_count_branch(known_bits+"1")))
    %>

    index_t get_count(scan_t s, int mnr)
    {
        return ${get_count_branch("")};
    }

    #define BIN_NR(key_arg) ((key_arg >> base_bit) & ${2**bits - 1})

""", strict_undefined=True)

# }}}

# {{{ scan helpers

RADIX_SORT_SCAN_PREAMBLE_TPL = Template(r"""//CL//
    scan_t scan_t_neutral()
    {
        scan_t result;
        %for mnr in range(2**bits):
            result.c${padded_bin(mnr, bits)} = 0;
        %endfor
        return result;
    }

    // considers bits (base_bit+bits-1, ..., base_bit)
    scan_t scan_t_from_value(
        key_t key,
        int base_bit,
        int i
    )
    {
        // extract relevant bit range
        key_t bin_nr = BIN_NR(key);

        dbg_printf(("i: %d key:%d bin_nr:%d\n", i, key, bin_nr));

        scan_t result;
        %for mnr in range(2**bits):
            result.c${padded_bin(mnr, bits)} = (bin_nr == ${mnr});
        %endfor

        return result;
    }

    scan_t scan_t_add(scan_t a, scan_t b, bool across_seg_boundary)
    {
        %for mnr in range(2**bits):
            <% field = "c"+padded_bin(mnr, bits) %>
            b.${field} = a.${field} + b.${field};
        %endfor

        return b;
    }
""", strict_undefined=True)

RADIX_SORT_OUTPUT_STMT_TPL = Template(r"""//CL//
    {
        key_t key = ${key_expr};
        key_t my_bin_nr = BIN_NR(key);

        index_t previous_bins_size = 0;
        %for mnr in range(2**bits):
            previous_bins_size +=
                (my_bin_nr > ${mnr})
                    ? last_item.c${padded_bin(mnr, bits)}
                    : 0;
        %endfor

        index_t tgt_idx =
            previous_bins_size
            + get_count(item, my_bin_nr) - 1;

        %for arg_name in sort_arg_names:
            sorted_${arg_name}[tgt_idx] = ${arg_name}[i];
        %endfor
    }
""", strict_undefined=True)

# }}}

# {{{ driver

class RadixSort(object):
    """Provides a general least-significant-digit `radix sort
    <https://en.wikipedia.org/wiki/Radix_sort>`_ on the compute device.
    """
    def __init__(self, context, arguments, key_expr, sort_arg_names,
            bits_at_a_time=4, index_dtype=np.int32, key_dtype=np.uint32,
            options=[]):
        """
        :arg arguments: A string of comma-separated C argument declarations.
            If *arguments* is specified, then *input_expr* must also be
            specified. All types used here must be known to PyOpenCL.
            (see :func:`pyopencl.tools.register_dtype`).
        :arg key_expr: An integer-valued C expression returning the
            key to which radix sort is applied. The array index
            for which the key is to be computed is available as `i`.
            The expression may refer to any of the *arguments*.
        :arg sort_arg_names: A list of argument names whose corresponding
            array arguments will be sorted according to *key_expr*.
        """

        # {{{ arg processing

        from pyopencl.scan import _parse_args
        self.arguments = _parse_args(arguments)
        del arguments

        self.sort_arg_names = sort_arg_names
        self.bits = int(bits_at_a_time)
        self.index_dtype = np.dtype(index_dtype)
        self.key_dtype = np.dtype(key_dtype)

        self.options = options

        # }}}

        # {{{ kernel creation

        scan_ctype, scan_dtype, scan_t_cdecl = \
                _make_sort_scan_type(context.devices[0], self.bits, self.index_dtype)

        from pyopencl.tools import VectorArg, ScalarArg
        scan_arguments = (
                list(self.arguments)
                + [VectorArg(arg.dtype, "sorted_"+arg.name) for arg in self.arguments
                    if arg.name in sort_arg_names]
                + [ ScalarArg(np.int32, "base_bit") ])

        codegen_args = dict(
                bits=self.bits,
                key_ctype=dtype_to_ctype(self.key_dtype),
                key_expr=key_expr,
                index_ctype=dtype_to_ctype(self.index_dtype),
                index_type_max=np.iinfo(self.index_dtype).max,
                padded_bin=_padded_bin,
                scan_ctype=scan_ctype,
                sort_arg_names=sort_arg_names,
                )

        preamble = scan_t_cdecl+RADIX_SORT_PREAMBLE_TPL.render(**codegen_args)
        scan_preamble = preamble + RADIX_SORT_SCAN_PREAMBLE_TPL.render(**codegen_args)

        from pyopencl.scan import GenericScanKernel
        self.scan_kernel = GenericScanKernel(
                context, scan_dtype,
                arguments=scan_arguments,
                input_expr="scan_t_from_value(%s, base_bit, i)" % key_expr,
                scan_expr="scan_t_add(a, b, across_seg_boundary)",
                neutral="scan_t_neutral()",
                output_statement=RADIX_SORT_OUTPUT_STMT_TPL.render(**codegen_args),
                preamble=scan_preamble, options=self.options)

        for i, arg in enumerate(self.arguments):
            if isinstance(arg, VectorArg):
                self.first_array_arg_idx = i

        # }}}

    def __call__(self, *args, **kwargs):
        """Run the radix sort. In addition to *args* which must match the
        *arguments* specification on the constructor, the following
        keyword arguments are supported:

        :arg key_bits: specify how many bits (starting from least-significant)
            there are in the key.
        :arg queue: A :class:`pyopencl.CommandQueue`, defaulting to the
            one from the first argument array.
        :arg allocator: See the *allocator* argument of :func:`pyopencl.array.empty`.
        :returns: Sorted copies of the arrays named in *sorted_args*, in the order
            of that list.
        """

        # {{{ run control

        key_bits = kwargs.pop("key_bits", None)
        if key_bits is None:
            key_bits = int(np.iinfo(self.key_dtype).bits)

        n = len(args[self.first_array_arg_idx])

        allocator = kwargs.pop("allocator", None)
        if allocator is None:
            allocator = args[self.first_array_arg_idx].allocator

        queue = kwargs.pop("allocator", None)
        if queue is None:
            queue = args[self.first_array_arg_idx].queue

        args = list(args)

        kwargs = dict(queue=queue)

        base_bit = 0
        while base_bit < key_bits:
            sorted_args = [
                    cl.array.empty(queue, n, arg_descr.dtype, allocator=allocator)
                    for arg_descr in self.arguments
                    if arg_descr.name in self.sort_arg_names]

            scan_args = args + sorted_args + [base_bit]

            self.scan_kernel(*scan_args, **kwargs)

            # substitute sorted
            for i, arg_descr in enumerate(self.arguments):
                if arg_descr.name in self.sort_arg_names:
                    args[i] = sorted_args[self.sort_arg_names.index(arg_descr.name)]

            base_bit += self.bits

        return [arg_val
                for arg_descr, arg_val in zip(self.arguments, args)
                if arg_descr.name in self.sort_arg_names]

        # }}}

# }}}

# }}}

# vim: filetype=pyopencl:fdm=marker
