"""Scan primitive."""

from __future__ import division
from __future__ import absolute_import
from six.moves import range
from six.moves import zip

__copyright__ = """Copyright 2011-2012 Andreas Kloeckner"""

__license__ = """
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
import pyopencl.array  # noqa
from pyopencl.scan import ScanTemplate
from pyopencl.tools import dtype_to_ctype
from pytools import memoize, memoize_method, Record
from mako.template import Template


# {{{ copy_if

_copy_if_template = ScanTemplate(
        arguments="item_t *ary, item_t *out, scan_t *count",
        input_expr="(%(predicate)s) ? 1 : 0",
        scan_expr="a+b", neutral="0",
        output_statement="""
            if (prev_item != item) out[item-1] = ary[i];
            if (i+1 == N) *count = item;
            """,
        template_processor="printf")


def extract_extra_args_types_values(extra_args):
    from pyopencl.tools import VectorArg, ScalarArg

    extra_args_types = []
    extra_args_values = []
    for name, val in extra_args:
        if isinstance(val, cl.array.Array):
            extra_args_types.append(VectorArg(val.dtype, name, with_offset=False))
            extra_args_values.append(val)
        elif isinstance(val, np.generic):
            extra_args_types.append(ScalarArg(val.dtype, name))
            extra_args_values.append(val)
        else:
            raise RuntimeError("argument '%d' not understood" % name)

    return tuple(extra_args_types), extra_args_values


def copy_if(ary, predicate, extra_args=[], preamble="", queue=None, wait_for=None):
    """Copy the elements of *ary* satisfying *predicate* to an output array.

    :arg predicate: a C expression evaluating to a `bool`, represented as a string.
        The value to test is available as `ary[i]`, and if the expression evaluates
        to `true`, then this value ends up in the output.
    :arg extra_args: |scan_extra_args|
    :arg preamble: |preamble|
    :arg wait_for: |explain-waitfor|
    :returns: a tuple *(out, count, event)* where *out* is the output array, *count*
        is an on-device scalar (fetch to host with `count.get()`) indicating
        how many elements satisfied *predicate*, and *event* is a
        :class:`pyopencl.Event` for dependency management. *out* is allocated
        to the same length as *ary*, but only the first *count* entries carry
        meaning.

    .. versionadded:: 2013.1
    """
    if len(ary) > np.iinfo(np.int32).max:
        scan_dtype = np.int64
    else:
        scan_dtype = np.int32

    extra_args_types, extra_args_values = extract_extra_args_types_values(extra_args)

    knl = _copy_if_template.build(ary.context,
            type_aliases=(("scan_t", scan_dtype), ("item_t", ary.dtype)),
            var_values=(("predicate", predicate),),
            more_preamble=preamble, more_arguments=extra_args_types)
    out = cl.array.empty_like(ary)
    count = ary._new_with_changes(data=None, offset=0,
            shape=(), strides=(), dtype=scan_dtype)

    # **dict is a Py2.5 workaround
    evt = knl(ary, out, count, *extra_args_values,
            **dict(queue=queue, wait_for=wait_for))

    return out, count, evt

# }}}


# {{{ remove_if

def remove_if(ary, predicate, extra_args=[], preamble="", queue=None, wait_for=None):
    """Copy the elements of *ary* not satisfying *predicate* to an output array.

    :arg predicate: a C expression evaluating to a `bool`, represented as a string.
        The value to test is available as `ary[i]`, and if the expression evaluates
        to `false`, then this value ends up in the output.
    :arg extra_args: |scan_extra_args|
    :arg preamble: |preamble|
    :arg wait_for: |explain-waitfor|
    :returns: a tuple *(out, count, event)* where *out* is the output array, *count*
        is an on-device scalar (fetch to host with `count.get()`) indicating
        how many elements did not satisfy *predicate*, and *event* is a
        :class:`pyopencl.Event` for dependency management.

    .. versionadded:: 2013.1
    """
    return copy_if(ary, "!(%s)" % predicate, extra_args=extra_args,
            preamble=preamble, queue=queue, wait_for=wait_for)

# }}}


# {{{ partition

_partition_template = ScanTemplate(
        arguments=(
            "item_t *ary, item_t *out_true, item_t *out_false, "
            "scan_t *count_true"),
        input_expr="(%(predicate)s) ? 1 : 0",
        scan_expr="a+b", neutral="0",
        output_statement="""//CL//
                if (prev_item != item)
                    out_true[item-1] = ary[i];
                else
                    out_false[i-item] = ary[i];
                if (i+1 == N) *count_true = item;
                """,
        template_processor="printf")


def partition(ary, predicate, extra_args=[], preamble="", queue=None, wait_for=None):
    """Copy the elements of *ary* into one of two arrays depending on whether
    they satisfy *predicate*.

    :arg predicate: a C expression evaluating to a `bool`, represented as a string.
        The value to test is available as `ary[i]`.
    :arg extra_args: |scan_extra_args|
    :arg preamble: |preamble|
    :arg wait_for: |explain-waitfor|
    :returns: a tuple *(out_true, out_false, count, event)* where *count*
        is an on-device scalar (fetch to host with `count.get()`) indicating
        how many elements satisfied the predicate, and *event* is a
        :class:`pyopencl.Event` for dependency management.

    .. versionadded:: 2013.1
    """
    if len(ary) > np.iinfo(np.uint32).max:
        scan_dtype = np.uint64
    else:
        scan_dtype = np.uint32

    extra_args_types, extra_args_values = extract_extra_args_types_values(extra_args)

    knl = _partition_template.build(
            ary.context,
            type_aliases=(("item_t", ary.dtype), ("scan_t", scan_dtype)),
            var_values=(("predicate", predicate),),
            more_preamble=preamble, more_arguments=extra_args_types)

    out_true = cl.array.empty_like(ary)
    out_false = cl.array.empty_like(ary)
    count = ary._new_with_changes(data=None, offset=0,
            shape=(), strides=(), dtype=scan_dtype)

    # **dict is a Py2.5 workaround
    evt = knl(ary, out_true, out_false, count, *extra_args_values,
            **dict(queue=queue, wait_for=wait_for))

    return out_true, out_false, count, evt

# }}}


# {{{ unique

_unique_template = ScanTemplate(
        arguments="item_t *ary, item_t *out, scan_t *count_unique",
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
        preamble="#define IS_EQUAL_EXPR(a, b) %(macro_is_equal_expr)s\n",
        template_processor="printf")


def unique(ary, is_equal_expr="a == b", extra_args=[], preamble="",
        queue=None, wait_for=None):
    """Copy the elements of *ary* into the output if *is_equal_expr*, applied to the
    array element and its predecessor, yields false.

    Works like the UNIX command :program:`uniq`, with a potentially custom
    comparison.  This operation is often used on sorted sequences.

    :arg is_equal_expr: a C expression evaluating to a `bool`,
        represented as a string.  The elements being compared are
        available as `a` and `b`. If this expression yields `false`, the
        two are considered distinct.
    :arg extra_args: |scan_extra_args|
    :arg preamble: |preamble|
    :arg wait_for: |explain-waitfor|
    :returns: a tuple *(out, count, event)* where *out* is the output array, *count*
        is an on-device scalar (fetch to host with `count.get()`) indicating
        how many elements satisfied the predicate, and *event* is a
        :class:`pyopencl.Event` for dependency management.

    .. versionadded:: 2013.1
    """

    if len(ary) > np.iinfo(np.uint32).max:
        scan_dtype = np.uint64
    else:
        scan_dtype = np.uint32

    extra_args_types, extra_args_values = extract_extra_args_types_values(extra_args)

    knl = _unique_template.build(
            ary.context,
            type_aliases=(("item_t", ary.dtype), ("scan_t", scan_dtype)),
            var_values=(("macro_is_equal_expr", is_equal_expr),),
            more_preamble=preamble, more_arguments=extra_args_types)

    out = cl.array.empty_like(ary)
    count = ary._new_with_changes(data=None, offset=0,
            shape=(), strides=(), dtype=scan_dtype)

    # **dict is a Py2.5 workaround
    evt = knl(ary, out, count, *extra_args_values,
            **dict(queue=queue, wait_for=wait_for))

    return out, count, evt

# }}}


# {{{ radix_sort

def to_bin(n):
    # Py 2.5 has no built-in bin()
    digs = []
    while n:
        digs.append(str(n % 2))
        n >>= 1

    return ''.join(digs[::-1])


def _padded_bin(i, l):
    s = to_bin(i)
    while len(s) < l:
        s = '0' + s
    return s


@memoize
def _make_sort_scan_type(device, bits, index_dtype):
    name = "pyopencl_sort_scan_%s_%dbits_t" % (
            index_dtype.type.__name__, bits)

    fields = []
    for mnr in range(2**bits):
        fields.append(('c%s' % _padded_bin(mnr, bits), index_dtype))

    dtype = np.dtype(fields)

    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct
    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)

    dtype = get_or_register_dtype(name, dtype)
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

# import hoisted here to be used as a default argument in the constructor
from pyopencl.scan import GenericScanKernel


class RadixSort(object):
    """Provides a general `radix sort <https://en.wikipedia.org/wiki/Radix_sort>`_
    on the compute device.

    .. seealso:: :class:`pyopencl.algorithm.BitonicSort`

    .. versionadded:: 2013.1
    """
    def __init__(self, context, arguments, key_expr, sort_arg_names,
            bits_at_a_time=2, index_dtype=np.int32, key_dtype=np.uint32,
            scan_kernel=GenericScanKernel, options=[]):
        """
        :arg arguments: A string of comma-separated C argument declarations.
            If *arguments* is specified, then *input_expr* must also be
            specified. All types used here must be known to PyOpenCL.
            (see :func:`pyopencl.tools.get_or_register_dtype`).
        :arg key_expr: An integer-valued C expression returning the
            key based on which the sort is performed. The array index
            for which the key is to be computed is available as `i`.
            The expression may refer to any of the *arguments*.
        :arg sort_arg_names: A list of argument names whose corresponding
            array arguments will be sorted according to *key_expr*.
        """

        # {{{ arg processing

        from pyopencl.tools import parse_arg_list
        self.arguments = parse_arg_list(arguments)
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
                + [ScalarArg(np.int32, "base_bit")])

        def get_count_branch(known_bits):
            if len(known_bits) == self.bits:
                return "s.c%s" % known_bits

            boundary_mnr = known_bits + "1" + (self.bits-len(known_bits)-1)*"0"

            return ("((mnr < %s) ? %s : %s)" % (
                int(boundary_mnr, 2),
                get_count_branch(known_bits+"0"),
                get_count_branch(known_bits+"1")))

        codegen_args = dict(
                bits=self.bits,
                key_ctype=dtype_to_ctype(self.key_dtype),
                key_expr=key_expr,
                index_ctype=dtype_to_ctype(self.index_dtype),
                index_type_max=np.iinfo(self.index_dtype).max,
                padded_bin=_padded_bin,
                scan_ctype=scan_ctype,
                sort_arg_names=sort_arg_names,
                get_count_branch=get_count_branch,
                )

        preamble = scan_t_cdecl+RADIX_SORT_PREAMBLE_TPL.render(**codegen_args)
        scan_preamble = preamble \
                + RADIX_SORT_SCAN_PREAMBLE_TPL.render(**codegen_args)

        self.scan_kernel = scan_kernel(
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
        :arg allocator: See the *allocator* argument of :func:`pyopencl.array.empty`.
        :arg queue: A :class:`pyopencl.CommandQueue`, defaulting to the
            one from the first argument array.
        :arg wait_for: |explain-waitfor|
        :returns: A tuple ``(sorted, event)``. *sorted* consists of sorted
            copies of the arrays named in *sorted_args*, in the order of that
            list. *event* is a :class:`pyopencl.Event` for dependency management.
        """

        wait_for = kwargs.pop("wait_for", None)

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

        base_bit = 0
        while base_bit < key_bits:
            sorted_args = [
                    cl.array.empty(queue, n, arg_descr.dtype, allocator=allocator)
                    for arg_descr in self.arguments
                    if arg_descr.name in self.sort_arg_names]

            scan_args = args + sorted_args + [base_bit]

            last_evt = self.scan_kernel(*scan_args,
                    **dict(queue=queue, wait_for=wait_for))
            wait_for = [last_evt]

            # substitute sorted
            for i, arg_descr in enumerate(self.arguments):
                if arg_descr.name in self.sort_arg_names:
                    args[i] = sorted_args[self.sort_arg_names.index(arg_descr.name)]

            base_bit += self.bits

        return [arg_val
                for arg_descr, arg_val in zip(self.arguments, args)
                if arg_descr.name in self.sort_arg_names], last_evt

        # }}}

# }}}

# }}}


# {{{ generic parallel list builder

# {{{ kernel template

_LIST_BUILDER_TEMPLATE = Template("""//CL//
% if double_support:
    #if __OPENCL_C_VERSION__ < 120
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
    #endif
    #define PYOPENCL_DEFINE_CDOUBLE
% endif

#include <pyopencl-complex.h>

${preamble}

// {{{ declare helper macros for user interface

typedef ${index_type} index_type;

%if is_count_stage:
    #define PLB_COUNT_STAGE

    %for name, dtype in list_names_and_dtypes:
        %if name in count_sharing:
            #define APPEND_${name}(value) { /* nothing */ }
        %else:
            #define APPEND_${name}(value) { ++(*plb_loc_${name}_count); }
        %endif
    %endfor
%else:
    #define PLB_WRITE_STAGE

    %for name, dtype in list_names_and_dtypes:
        %if name in count_sharing:
            #define APPEND_${name}(value) \
                { plb_${name}_list[(*plb_${count_sharing[name]}_index) - 1] \
                    = value; }
        %else:
            #define APPEND_${name}(value) \
                { plb_${name}_list[(*plb_${name}_index)++] = value; }
        %endif
    %endfor
%endif

#define LIST_ARG_DECL ${user_list_arg_decl}
#define LIST_ARGS ${user_list_args}
#define USER_ARG_DECL ${user_arg_decl}
#define USER_ARGS ${user_args}

// }}}

${generate_template}

// {{{ kernel entry point

__kernel
%if do_not_vectorize:
__attribute__((reqd_work_group_size(1, 1, 1)))
%endif
void ${kernel_name}(${kernel_list_arg_decl} USER_ARG_DECL index_type n)

{
    %if not do_not_vectorize:
        int lid = get_local_id(0);
        index_type gsize = get_global_size(0);
        index_type work_group_start = get_local_size(0)*get_group_id(0);
        for (index_type i = work_group_start + lid; i < n; i += gsize)
    %else:
        const int chunk_size = 128;
        index_type chunk_base = get_global_id(0)*chunk_size;
        index_type gsize = get_global_size(0);
        for (; chunk_base < n; chunk_base += gsize*chunk_size)
        for (index_type i = chunk_base; i < min(n, chunk_base+chunk_size); ++i)
    %endif
    {
        %if is_count_stage:
            %for name, dtype in list_names_and_dtypes:
                %if name not in count_sharing:
                    index_type plb_loc_${name}_count = 0;
                %endif
            %endfor
        %else:
            %for name, dtype in list_names_and_dtypes:
                %if name not in count_sharing:
                    index_type plb_${name}_index =
                        plb_${name}_start_index[i];
                %endif
            %endfor
        %endif

        generate(${kernel_list_arg_values} USER_ARGS i);

        %if is_count_stage:
            %for name, dtype in list_names_and_dtypes:
                %if name not in count_sharing:
                    plb_${name}_count[i] = plb_loc_${name}_count;
                %endif
            %endfor
        %endif
    }
}

// }}}

""", strict_undefined=True)

# }}}


def _get_arg_decl(arg_list):
    result = ""
    for arg in arg_list:
        result += arg.declarator() + ", "

    return result


def _get_arg_list(arg_list, prefix=""):
    result = ""
    for arg in arg_list:
        result += prefix + arg.name + ", "

    return result


class BuiltList(Record):
    pass


class ListOfListsBuilder:
    """Generates and executes code to produce a large number of variable-size
    lists, simply.

    .. note:: This functionality is provided as a preview. Its interface
        is subject to change until this notice is removed.

    .. versionadded:: 2013.1

    Here's a usage example::

        from pyopencl.algorithm import ListOfListsBuilder
        builder = ListOfListsBuilder(context, [("mylist", np.int32)], \"\"\"
                void generate(LIST_ARG_DECL USER_ARG_DECL index_type i)
                {
                    int count = i % 4;
                    for (int j = 0; j < count; ++j)
                    {
                        APPEND_mylist(count);
                    }
                }
                \"\"\", arg_decls=[])

        result, event = builder(queue, 2000)

        inf = result["mylist"]
        assert inf.count == 3000
        assert (inf.list.get()[-6:] == [1, 2, 2, 3, 3, 3]).all()

    The function `generate` above is called once for each "input object".
    Each input object can then generate zero or more list entries.
    The number of these input objects is given to :meth:`__call__` as *n_objects*.
    List entries are generated by calls to `APPEND_<list name>(value)`.
    Multiple lists may be generated at once.

    """
    def __init__(self, context, list_names_and_dtypes, generate_template,
            arg_decls, count_sharing=None, devices=None,
            name_prefix="plb_build_list", options=[], preamble="",
            debug=False, complex_kernel=False):
        """
        :arg context: A :class:`pyopencl.Context`.
        :arg list_names_and_dtypes: a list of `(name, dtype)` tuples
            indicating the lists to be built.
        :arg generate_template: a snippet of C as described below
        :arg arg_decls: A string of comma-separated C argument declarations.
        :arg count_sharing: A mapping consisting of `(child, mother)`
            indicating that `mother` and `child` will always have the
            same number of indices, and the `APPEND` to `mother`
            will always happen *before* the `APPEND` to the child.
        :arg name_prefix: the name prefix to use for the compiled kernels
        :arg options: OpenCL compilation options for kernels using
            *generate_template*.
        :arg complex_kernel: If `True`, prevents vectorization on CPUs.

        *generate_template* may use the following C macros/identifiers:

        * `index_type`: expands to C identifier for the index type used
          for the calculation
        * `USER_ARG_DECL`: expands to the C declarator for `arg_decls`
        * `USER_ARGS`: a list of C argument values corresponding to
          `user_arg_decl`
        * `LIST_ARG_DECL`: expands to a C argument list representing the
          data for the output lists. These are escaped prefixed with
          `"plg_"` so as to not interfere with user-provided names.
        * `LIST_ARGS`: a list of C argument values corresponding to
          `LIST_ARG_DECL`
        * `APPEND_name(entry)`: inserts `entry` into the list `name`.
          *entry* must be a valid C expression of the correct type.

        All argument-list related macros have a trailing comma included
        if they are non-empty.

        *generate_template* must supply a function:

        .. code-block:: c

            void generate(USER_ARG_DECL LIST_ARG_DECL index_type i)
            {
                APPEND_mylist(5);
            }

        Internally, the `kernel_template` is expanded (at least) twice. Once,
        for a 'counting' stage where the size of all the lists is determined,
        and a second time, for a 'generation' stage where the lists are
        actually filled. A `generate` function that has side effects beyond
        calling `append` is therefore ill-formed.
        """

        if devices is None:
            devices = context.devices

        if count_sharing is None:
            count_sharing = {}

        self.context = context
        self.devices = devices

        self.list_names_and_dtypes = list_names_and_dtypes
        self.generate_template = generate_template

        from pyopencl.tools import parse_arg_list
        self.arg_decls = parse_arg_list(arg_decls)

        self.count_sharing = count_sharing

        self.name_prefix = name_prefix
        self.preamble = preamble
        self.options = options

        self.debug = debug

        self.complex_kernel = complex_kernel

    # {{{ kernel generators

    @memoize_method
    def get_scan_kernel(self, index_dtype):
        from pyopencl.scan import GenericScanKernel
        return GenericScanKernel(
                self.context, index_dtype,
                arguments="__global %s *ary" % dtype_to_ctype(index_dtype),
                input_expr="ary[i]",
                scan_expr="a+b", neutral="0",
                output_statement="ary[i+1] = item;",
                devices=self.devices)

    def do_not_vectorize(self):
        from pytools import any
        return (self.complex_kernel
                and any(dev.type & cl.device_type.CPU
                    for dev in self.context.devices))

    @memoize_method
    def get_count_kernel(self, index_dtype):
        index_ctype = dtype_to_ctype(index_dtype)
        from pyopencl.tools import VectorArg, OtherArg
        kernel_list_args = [
                VectorArg(index_dtype, "plb_%s_count" % name)
                for name, dtype in self.list_names_and_dtypes
                if name not in self.count_sharing]

        user_list_args = []
        for name, dtype in self.list_names_and_dtypes:
            if name in self.count_sharing:
                continue

            name = "plb_loc_%s_count" % name
            user_list_args.append(OtherArg("%s *%s" % (
                index_ctype, name), name))

        kernel_name = self.name_prefix+"_count"

        from pyopencl.characterize import has_double_support
        src = _LIST_BUILDER_TEMPLATE.render(
                is_count_stage=True,
                kernel_name=kernel_name,
                double_support=all(has_double_support(dev) for dev in
                    self.context.devices),
                debug=self.debug,
                do_not_vectorize=self.do_not_vectorize(),

                kernel_list_arg_decl=_get_arg_decl(kernel_list_args),
                kernel_list_arg_values=_get_arg_list(user_list_args, prefix="&"),
                user_list_arg_decl=_get_arg_decl(user_list_args),
                user_list_args=_get_arg_list(user_list_args),
                user_arg_decl=_get_arg_decl(self.arg_decls),
                user_args=_get_arg_list(self.arg_decls),

                list_names_and_dtypes=self.list_names_and_dtypes,
                count_sharing=self.count_sharing,
                name_prefix=self.name_prefix,
                generate_template=self.generate_template,
                preamble=self.preamble,

                index_type=index_ctype,
                )

        src = str(src)

        prg = cl.Program(self.context, src).build(self.options)
        knl = getattr(prg, kernel_name)

        from pyopencl.tools import get_arg_list_scalar_arg_dtypes
        knl.set_scalar_arg_dtypes(get_arg_list_scalar_arg_dtypes(
            kernel_list_args+self.arg_decls) + [index_dtype])

        return knl

    @memoize_method
    def get_write_kernel(self, index_dtype):
        index_ctype = dtype_to_ctype(index_dtype)
        from pyopencl.tools import VectorArg, OtherArg
        kernel_list_args = []
        kernel_list_arg_values = ""
        user_list_args = []

        for name, dtype in self.list_names_and_dtypes:
            list_name = "plb_%s_list" % name
            list_arg = VectorArg(dtype, list_name)

            kernel_list_args.append(list_arg)
            user_list_args.append(list_arg)

            if name in self.count_sharing:
                kernel_list_arg_values += "%s, " % list_name
                continue

            kernel_list_args.append(
                    VectorArg(index_dtype, "plb_%s_start_index" % name))

            index_name = "plb_%s_index" % name
            user_list_args.append(OtherArg("%s *%s" % (
                index_ctype, index_name), index_name))

            kernel_list_arg_values += "%s, &%s, " % (list_name, index_name)

        kernel_name = self.name_prefix+"_write"

        from pyopencl.characterize import has_double_support
        src = _LIST_BUILDER_TEMPLATE.render(
                is_count_stage=False,
                kernel_name=kernel_name,
                double_support=all(has_double_support(dev) for dev in
                    self.context.devices),
                debug=self.debug,
                do_not_vectorize=self.do_not_vectorize(),

                kernel_list_arg_decl=_get_arg_decl(kernel_list_args),
                kernel_list_arg_values=kernel_list_arg_values,
                user_list_arg_decl=_get_arg_decl(user_list_args),
                user_list_args=_get_arg_list(user_list_args),
                user_arg_decl=_get_arg_decl(self.arg_decls),
                user_args=_get_arg_list(self.arg_decls),

                list_names_and_dtypes=self.list_names_and_dtypes,
                count_sharing=self.count_sharing,
                name_prefix=self.name_prefix,
                generate_template=self.generate_template,
                preamble=self.preamble,

                index_type=index_ctype,
                )

        src = str(src)

        prg = cl.Program(self.context, src).build(self.options)
        knl = getattr(prg, kernel_name)

        from pyopencl.tools import get_arg_list_scalar_arg_dtypes
        knl.set_scalar_arg_dtypes(get_arg_list_scalar_arg_dtypes(
            kernel_list_args+self.arg_decls) + [index_dtype])

        return knl

    # }}}

    # {{{ driver

    def __call__(self, queue, n_objects, *args, **kwargs):
        """
        :arg args: arguments corresponding to arg_decls in the constructor.
            :class:`pyopencl.array.Array` are not allowed directly and should
            be passed as their :attr:`pyopencl.array.Array.data` attribute instead.
        :arg allocator: optionally, the allocator to use to allocate new
            arrays.
        :arg wait_for: |explain-waitfor|
        :returns: a tuple ``(lists, event)``, where
            *lists* a mapping from (built) list names to objects which
            have attributes

            * ``count`` for the total number of entries in all lists combined
            * ``lists`` for the array containing all lists.
            * ``starts`` for the array of starting indices in `lists`.
              `starts` is built so that it has n+1 entries, so that
              the *i*'th entry is the start of the *i*'th list, and the
              *i*'th entry is the index one past the *i*'th list's end,
              even for the last list.

              This implies that all lists are contiguous.

              *event* is a :class:`pyopencl.Event` for dependency management.
        """
        if n_objects >= int(np.iinfo(np.int32).max):
            index_dtype = np.int64
        else:
            index_dtype = np.int32
        index_dtype = np.dtype(index_dtype)

        allocator = kwargs.pop("allocator", None)
        wait_for = kwargs.pop("wait_for", None)
        if kwargs:
            raise TypeError("invalid keyword arguments: '%s'" % ", ".join(kwargs))

        result = {}
        count_list_args = []

        if wait_for is None:
            wait_for = []

        count_kernel = self.get_count_kernel(index_dtype)
        write_kernel = self.get_write_kernel(index_dtype)
        scan_kernel = self.get_scan_kernel(index_dtype)

        # {{{ allocate memory for counts

        for name, dtype in self.list_names_and_dtypes:
            if name in self.count_sharing:
                continue

            counts = cl.array.empty(queue,
                    (n_objects + 1), index_dtype, allocator=allocator)
            counts[-1] = 0
            wait_for = wait_for + counts.events

            # The scan will turn the "counts" array into the "starts" array
            # in-place.
            result[name] = BuiltList(starts=counts)
            count_list_args.append(counts.data)

        # }}}

        if self.debug:
            gsize = (1,)
            lsize = (1,)
        elif self.complex_kernel and queue.device.type == cl.device_type.CPU:
            gsize = (4*queue.device.max_compute_units,)
            lsize = (1,)
        else:
            from pyopencl.array import splay
            gsize, lsize = splay(queue, n_objects)

        count_event = count_kernel(queue, gsize, lsize,
                *(tuple(count_list_args) + args + (n_objects,)),
                **dict(wait_for=wait_for))

        # {{{ run scans

        scan_events = []

        for name, dtype in self.list_names_and_dtypes:
            if name in self.count_sharing:
                continue

            info_record = result[name]
            starts_ary = info_record.starts
            evt = scan_kernel(starts_ary, wait_for=[count_event],
                    size=n_objects)

            starts_ary.setitem(0, 0, queue=queue, wait_for=[evt])
            scan_events.extend(starts_ary.events)

            # retrieve count
            info_record.count = int(starts_ary[-1].get())

        # }}}

        # {{{ deal with count-sharing lists, allocate memory for lists

        write_list_args = []
        for name, dtype in self.list_names_and_dtypes:
            if name in self.count_sharing:
                sharing_from = self.count_sharing[name]

                info_record = result[name] = BuiltList(
                        count=result[sharing_from].count,
                        starts=result[sharing_from].starts,
                        )

            else:
                info_record = result[name]

            info_record.lists = cl.array.empty(queue,
                    info_record.count, dtype, allocator=allocator)
            write_list_args.append(info_record.lists.data)

            if name not in self.count_sharing:
                write_list_args.append(info_record.starts.data)

        # }}}

        evt = write_kernel(queue, gsize, lsize,
                *(tuple(write_list_args) + args + (n_objects,)),
                **dict(wait_for=scan_events))

        return result, evt

    # }}}

# }}}


# {{{ key-value sorting

class _KernelInfo(Record):
    pass


def _make_cl_int_literal(value, dtype):
    iinfo = np.iinfo(dtype)
    result = str(int(value))
    if dtype.itemsize == 8:
        result += "l"
    if int(iinfo.min) < 0:
        result += "u"

    return result


class KeyValueSorter(object):
    """Given arrays *values* and *keys* of equal length
    and a number *nkeys* of keys, returns a tuple `(starts,
    lists)`, as follows: *values* and *keys* are sorted
    by *keys*, and the sorted *values* is returned as
    *lists*. Then for each index *i* in `range(nkeys)`,
    *starts[i]* is written to indicating where the
    group of *values* belonging to the key with index
    *i* begins. It implicitly ends at *starts[i+1]*.

    `starts` is built so that it has `nkeys+1` entries, so that
    the *i*'th entry is the start of the *i*'th list, and the
    *i*'th entry is the index one past the *i*'th list's end,
    even for the last list.

    This implies that all lists are contiguous.

    .. note:: This functionality is provided as a preview. Its
        interface is subject to change until this notice is removed.

    .. versionadded:: 2013.1
    """

    def __init__(self, context):
        self.context = context

    @memoize_method
    def get_kernels(self, key_dtype, value_dtype, starts_dtype):
        from pyopencl.algorithm import RadixSort
        from pyopencl.tools import VectorArg, ScalarArg

        by_target_sorter = RadixSort(
                self.context, [
                    VectorArg(value_dtype, "values"),
                    VectorArg(key_dtype, "keys"),
                    ],
                key_expr="keys[i]",
                sort_arg_names=["values", "keys"])

        from pyopencl.elementwise import ElementwiseTemplate
        start_finder = ElementwiseTemplate(
                arguments="""//CL//
                starts_t *key_group_starts,
                key_t *keys_sorted_by_key,
                """,

                operation=r"""//CL//
                key_t my_key = keys_sorted_by_key[i];

                if (i == 0 || my_key != keys_sorted_by_key[i-1])
                    key_group_starts[my_key] = i;
                """,
                name="find_starts").build(self.context,
                        type_aliases=(
                            ("key_t", starts_dtype),
                            ("starts_t", starts_dtype),
                            ),
                        var_values=())

        from pyopencl.scan import GenericScanKernel
        bound_propagation_scan = GenericScanKernel(
                self.context, starts_dtype,
                arguments=[
                    VectorArg(starts_dtype, "starts"),
                    # starts has length n+1
                    ScalarArg(key_dtype, "nkeys"),
                    ],
                input_expr="starts[nkeys-i]",
                scan_expr="min(a, b)",
                neutral=_make_cl_int_literal(
                    np.iinfo(starts_dtype).max, starts_dtype),
                output_statement="starts[nkeys-i] = item;")

        return _KernelInfo(
                by_target_sorter=by_target_sorter,
                start_finder=start_finder,
                bound_propagation_scan=bound_propagation_scan)

    def __call__(self, queue, keys, values, nkeys,
            starts_dtype, allocator=None, wait_for=None):
        if allocator is None:
            allocator = values.allocator

        knl_info = self.get_kernels(keys.dtype, values.dtype,
                starts_dtype)

        (values_sorted_by_key, keys_sorted_by_key), evt = knl_info.by_target_sorter(
                values, keys, queue=queue, wait_for=wait_for)

        starts = (cl.array.empty(queue, (nkeys+1), starts_dtype, allocator=allocator)
                .fill(len(values_sorted_by_key), wait_for=[evt]))
        evt, = starts.events

        evt = knl_info.start_finder(starts, keys_sorted_by_key,
                range=slice(len(keys_sorted_by_key)),
                wait_for=[evt])

        evt = knl_info.bound_propagation_scan(starts, nkeys,
                queue=queue, wait_for=[evt])

        return starts, values_sorted_by_key, evt

# }}}

# vim: filetype=pyopencl:fdm=marker
