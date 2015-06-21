from __future__ import division, with_statement
from __future__ import absolute_import
from __future__ import print_function
import six
from six.moves import range

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import cgen
import numpy as np
import re
from pymbolic.parser import Parser as ExpressionParserBase
from pymbolic.mapper import CombineMapper
import pymbolic.primitives as p
from pymbolic.mapper.c_code import CCodeMapper as CCodeMapperBase

from warnings import warn

import pytools.lex


class TranslatorWarning(UserWarning):
    pass


class TranslationError(RuntimeError):
    pass


def complex_type_name(dtype):
    if dtype == np.complex64:
        return "cfloat"
    if dtype == np.complex128:
        return "cdouble"
    else:
        raise RuntimeError


# {{{ AST components

def dtype_to_ctype(dtype):
    if dtype is None:
        raise ValueError("dtype may not be None")

    dtype = np.dtype(dtype)
    if dtype == np.int64:
        return "long"
    elif dtype == np.uint64:
        return "unsigned long"
    elif dtype == np.int32:
        return "int"
    elif dtype == np.uint32:
        return "unsigned int"
    elif dtype == np.int16:
        return "short int"
    elif dtype == np.uint16:
        return "short unsigned int"
    elif dtype == np.int8:
        return "signed char"
    elif dtype == np.uint8:
        return "unsigned char"
    elif dtype == np.float32:
        return "float"
    elif dtype == np.float64:
        return "double"
    elif dtype == np.complex64:
        return "cfloat_t"
    elif dtype == np.complex128:
        return "cdouble_t"
    else:
        raise ValueError("unable to map dtype '%s'" % dtype)


class POD(cgen.POD):
    def get_decl_pair(self):
        return [dtype_to_ctype(self.dtype)], self.name

# }}}


# {{{ expression parser

_less_than = intern("less_than")
_greater_than = intern("greater_than")
_less_equal = intern("less_equal")
_greater_equal = intern("greater_equal")
_equal = intern("equal")
_not_equal = intern("not_equal")

_not = intern("not")
_and = intern("and")
_or = intern("or")


class TypedLiteral(p.Leaf):
    def __init__(self, value, dtype):
        self.value = value
        self.dtype = np.dtype(dtype)

    def __getinitargs__(self):
        return self.value, self.dtype

    mapper_method = intern("map_literal")


def simplify_typed_literal(expr):
    if (isinstance(expr, p.Product)
            and len(expr.children) == 2
            and isinstance(expr.children[1], TypedLiteral)
            and p.is_constant(expr.children[0])
            and expr.children[0] == -1):
        tl = expr.children[1]
        return TypedLiteral("-"+tl.value, tl.dtype)
    else:
        return expr


class FortranExpressionParser(ExpressionParserBase):
    # FIXME double/single prec literals

    lex_table = [
        (_less_than, pytools.lex.RE(r"\.lt\.", re.I)),
        (_greater_than, pytools.lex.RE(r"\.gt\.", re.I)),
        (_less_equal, pytools.lex.RE(r"\.le\.", re.I)),
        (_greater_equal, pytools.lex.RE(r"\.ge\.", re.I)),
        (_equal, pytools.lex.RE(r"\.eq\.", re.I)),
        (_not_equal, pytools.lex.RE(r"\.ne\.", re.I)),

        (_not, pytools.lex.RE(r"\.not\.", re.I)),
        (_and, pytools.lex.RE(r"\.and\.", re.I)),
        (_or, pytools.lex.RE(r"\.or\.", re.I)),
        ] + ExpressionParserBase.lex_table

    def __init__(self, tree_walker):
        self.tree_walker = tree_walker

    _PREC_FUNC_ARGS = 1

    def parse_terminal(self, pstate):
        scope = self.tree_walker.scope_stack[-1]

        from pymbolic.parser import (
            _identifier, _openpar, _closepar, _float)

        next_tag = pstate.next_tag()
        if next_tag is _float:
            value = pstate.next_str_and_advance().lower()
            if "d" in value:
                dtype = np.float64
            else:
                dtype = np.float32

            value = value.replace("d", "e")
            if value.startswith("."):
                prev_value = value
                value = "0"+value
                print(value, prev_value)
            elif value.startswith("-."):
                prev_value = value
                value = "-0"+value[1:]
                print(value, prev_value)
            return TypedLiteral(value, dtype)

        elif next_tag is _identifier:
            name = pstate.next_str_and_advance()

            if pstate.is_at_end() or pstate.next_tag() is not _openpar:
                # not a subscript
                scope.use_name(name)

                return p.Variable(name)

            left_exp = p.Variable(name)

            pstate.advance()
            pstate.expect_not_end()

            if scope.is_known(name):
                cls = p.Subscript
            else:
                cls = p.Call

            if pstate.next_tag is _closepar:
                pstate.advance()
                left_exp = cls(left_exp, ())
            else:
                args = self.parse_expression(pstate, self._PREC_FUNC_ARGS)
                if not isinstance(args, tuple):
                    args = (args,)
                left_exp = cls(left_exp, args)
                pstate.expect(_closepar)
                pstate.advance()

            return left_exp
        else:
            return ExpressionParserBase.parse_terminal(
                    self, pstate)

    COMP_MAP = {
            _less_than: "<",
            _less_equal: "<=",
            _greater_than: ">",
            _greater_equal: ">=",
            _equal: "==",
            _not_equal: "!=",
            }

    def parse_prefix(self, pstate, min_precedence=0):
        from pymbolic.parser import _PREC_UNARY
        import pymbolic.primitives as primitives

        pstate.expect_not_end()

        if pstate.is_next(_not):
            pstate.advance()
            return primitives.LogicalNot(
                    self.parse_expression(pstate, _PREC_UNARY))
        else:
            return ExpressionParserBase.parse_prefix(self, pstate)

    def parse_postfix(self, pstate, min_precedence, left_exp):
        from pymbolic.parser import (
                _PREC_CALL, _PREC_COMPARISON, _openpar,
                _PREC_LOGICAL_OR, _PREC_LOGICAL_AND)
        from pymbolic.primitives import (
                Comparison, LogicalAnd, LogicalOr)

        next_tag = pstate.next_tag()
        if next_tag is _openpar and _PREC_CALL > min_precedence:
            raise TranslationError("parenthesis operator only works on names")
        elif next_tag in self.COMP_MAP and _PREC_COMPARISON > min_precedence:
            pstate.advance()
            left_exp = Comparison(
                    left_exp,
                    self.COMP_MAP[next_tag],
                    self.parse_expression(pstate, _PREC_COMPARISON))
            did_something = True
        elif next_tag is _and and _PREC_LOGICAL_AND > min_precedence:
            pstate.advance()
            left_exp = LogicalAnd((left_exp,
                    self.parse_expression(pstate, _PREC_LOGICAL_AND)))
            did_something = True
        elif next_tag is _or and _PREC_LOGICAL_OR > min_precedence:
            pstate.advance()
            left_exp = LogicalOr((left_exp,
                    self.parse_expression(pstate, _PREC_LOGICAL_OR)))
            did_something = True
        else:
            left_exp, did_something = ExpressionParserBase.parse_postfix(
                    self, pstate, min_precedence, left_exp)

            if isinstance(left_exp, tuple) and min_precedence < self._PREC_FUNC_ARGS:
                # this must be a complex literal
                assert len(left_exp) == 2
                r, i = left_exp

                r = simplify_typed_literal(r)
                i = simplify_typed_literal(i)

                dtype = (r.dtype.type(0) + i.dtype.type(0)).dtype
                if dtype == np.float32:
                    dtype = np.complex64
                else:
                    dtype = np.complex128

                left_exp = TypedLiteral(left_exp, dtype)

        return left_exp, did_something

# }}}


# {{{ expression generator

class TypeInferenceMapper(CombineMapper):
    def __init__(self, scope):
        self.scope = scope

    def combine(self, dtypes):
        return sum(dtype.type(1) for dtype in dtypes).dtype

    def map_literal(self, expr):
        return expr.dtype

    def map_constant(self, expr):
        return np.asarray(expr).dtype

    def map_variable(self, expr):
        return self.scope.get_type(expr.name)

    def map_call(self, expr):
        name = expr.function.name
        if name == "fromreal":
            arg, = expr.parameters
            base_dtype = self.rec(arg)
            tgt_real_dtype = (np.float32(0)+base_dtype.type(0)).dtype
            assert tgt_real_dtype.kind == "f"
            if tgt_real_dtype == np.float32:
                return np.dtype(np.complex64)
            elif tgt_real_dtype == np.float64:
                return np.dtype(np.complex128)
            else:
                raise RuntimeError("unexpected complex type")

        elif name in ["imag", "real", "abs", "dble"]:
            arg, = expr.parameters
            base_dtype = self.rec(arg)

            if base_dtype == np.complex128:
                return np.dtype(np.float64)
            elif base_dtype == np.complex64:
                return np.dtype(np.float32)
            else:
                return base_dtype

        else:
            return CombineMapper.map_call(self, expr)


class ComplexCCodeMapper(CCodeMapperBase):
    def __init__(self, infer_type):
        CCodeMapperBase.__init__(self)
        self.infer_type = infer_type

    def map_sum(self, expr, enclosing_prec):
        tgt_dtype = self.infer_type(expr)
        is_complex = tgt_dtype.kind == 'c'

        if not is_complex:
            return CCodeMapperBase.map_sum(self, expr, enclosing_prec)
        else:
            tgt_name = complex_type_name(tgt_dtype)

            reals = [child for child in expr.children
                    if 'c' != self.infer_type(child).kind]
            complexes = [child for child in expr.children
                    if 'c' == self.infer_type(child).kind]

            from pymbolic.mapper.stringifier import PREC_SUM, PREC_NONE
            real_sum = self.join_rec(" + ", reals, PREC_SUM)

            if len(complexes) == 1:
                myprec = PREC_SUM
            else:
                myprec = PREC_NONE

            complex_sum = self.rec(complexes[0], myprec)
            for child in complexes[1:]:
                complex_sum = "%s_add(%s, %s)" % (
                        tgt_name, complex_sum,
                        self.rec(child, PREC_NONE))

            if real_sum:
                result = "%s_add(%s_fromreal(%s), %s)" % (
                        tgt_name, tgt_name, real_sum, complex_sum)
            else:
                result = complex_sum

            return self.parenthesize_if_needed(result, enclosing_prec, PREC_SUM)

    def map_product(self, expr, enclosing_prec):
        tgt_dtype = self.infer_type(expr)
        is_complex = 'c' == tgt_dtype.kind

        if not is_complex:
            return CCodeMapperBase.map_product(self, expr, enclosing_prec)
        else:
            tgt_name = complex_type_name(tgt_dtype)

            reals = [child for child in expr.children
                    if 'c' != self.infer_type(child).kind]
            complexes = [child for child in expr.children
                    if 'c' == self.infer_type(child).kind]

            from pymbolic.mapper.stringifier import PREC_PRODUCT, PREC_NONE
            real_prd = self.join_rec("*", reals, PREC_PRODUCT)

            if len(complexes) == 1:
                myprec = PREC_PRODUCT
            else:
                myprec = PREC_NONE

            complex_prd = self.rec(complexes[0], myprec)
            for child in complexes[1:]:
                complex_prd = "%s_mul(%s, %s)" % (
                        tgt_name, complex_prd,
                        self.rec(child, PREC_NONE))

            if real_prd:
                result = "%s_rmul(%s, %s)" % (tgt_name, real_prd, complex_prd)
            else:
                result = complex_prd

            return self.parenthesize_if_needed(result, enclosing_prec, PREC_PRODUCT)

    def map_quotient(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        n_complex = 'c' == self.infer_type(expr.numerator).kind
        d_complex = 'c' == self.infer_type(expr.denominator).kind

        tgt_dtype = self.infer_type(expr)

        if not (n_complex or d_complex):
            return CCodeMapperBase.map_quotient(self, expr, enclosing_prec)
        elif n_complex and not d_complex:
            return "%s_divider(%s, %s)" % (
                    complex_type_name(tgt_dtype),
                    self.rec(expr.numerator, PREC_NONE),
                    self.rec(expr.denominator, PREC_NONE))
        elif not n_complex and d_complex:
            return "%s_rdivide(%s, %s)" % (
                    complex_type_name(tgt_dtype),
                    self.rec(expr.numerator, PREC_NONE),
                    self.rec(expr.denominator, PREC_NONE))
        else:
            return "%s_divide(%s, %s)" % (
                    complex_type_name(tgt_dtype),
                    self.rec(expr.numerator, PREC_NONE),
                    self.rec(expr.denominator, PREC_NONE))

    def map_remainder(self, expr, enclosing_prec):
        tgt_dtype = self.infer_type(expr)
        if 'c' == tgt_dtype.kind:
            raise RuntimeError("complex remainder not defined")

        return CCodeMapperBase.map_remainder(self, expr, enclosing_prec)

    def map_power(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE

        tgt_dtype = self.infer_type(expr)
        if 'c' == tgt_dtype.kind:
            if expr.exponent in [2, 3, 4]:
                value = expr.base
                for i in range(expr.exponent-1):
                    value = value * expr.base
                return self.rec(value, enclosing_prec)
            else:
                b_complex = 'c' == self.infer_type(expr.base).kind
                e_complex = 'c' == self.infer_type(expr.exponent).kind

                if b_complex and not e_complex:
                    return "%s_powr(%s, %s)" % (
                            complex_type_name(tgt_dtype),
                            self.rec(expr.base, PREC_NONE),
                            self.rec(expr.exponent, PREC_NONE))
                else:
                    return "%s_pow(%s, %s)" % (
                            complex_type_name(tgt_dtype),
                            self.rec(expr.base, PREC_NONE),
                            self.rec(expr.exponent, PREC_NONE))

        return CCodeMapperBase.map_power(self, expr, enclosing_prec)


class CCodeMapper(ComplexCCodeMapper):
    # Whatever is needed to mop up after Fortran goes here.
    # Stuff that deals with generating real-valued code
    # from complex code goes above.

    def __init__(self, translator, scope):
        ComplexCCodeMapper.__init__(self, scope.get_type_inference_mapper())
        self.translator = translator
        self.scope = scope

    def map_subscript(self, expr, enclosing_prec):
        idx_dtype = self.infer_type(expr.index)
        if not 'i' == idx_dtype.kind or 'u' == idx_dtype.kind:
            ind_prefix = "(int) "
        else:
            ind_prefix = ""

        idx = expr.index
        if isinstance(idx, tuple) and len(idx) == 1:
            idx, = idx

        from pymbolic.mapper.stringifier import PREC_NONE, PREC_CALL
        return self.parenthesize_if_needed(
                self.format("%s[%s%s]",
                    self.scope.translate_var_name(expr.aggregate.name),
                    ind_prefix,
                    self.rec(idx, PREC_NONE)),
                enclosing_prec, PREC_CALL)

    def map_call(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE

        tgt_dtype = self.infer_type(expr)
        arg_dtypes = [self.infer_type(par) for par in expr.parameters]

        name = expr.function.name
        if 'f' == tgt_dtype.kind and name == "abs":
            name = "fabs"

        elif 'c' == tgt_dtype.kind:
            if name in ["conjg", "dconjg"]:
                name = "conj"

            if name[:2] == "cd" and name[2:] in ["log", "exp", "sqrt"]:
                name = name[2:]

            if name == "dble":
                name = "real"

            name = "%s_%s" % (
                    complex_type_name(tgt_dtype),
                    name)

        elif name in ["aimag", "real", "imag"] and tgt_dtype.kind == "f":
            arg_dtype, = arg_dtypes

            if name == "aimag":
                name = "imag"

            name = "%s_%s" % (
                    complex_type_name(arg_dtype),
                    name)

        elif 'c' == tgt_dtype.kind and name == "abs":
            arg_dtype, = arg_dtypes

            name = "%s_abs" % (
                    complex_type_name(arg_dtype))

        return self.format("%s(%s)",
                name,
                self.join_rec(", ", expr.parameters, PREC_NONE))

    def map_variable(self, expr, enclosing_prec):
        # guaranteed to not be a subscript or a call

        name = expr.name
        shape = self.scope.get_shape(name)
        name = self.scope.translate_var_name(name)
        if expr.name in self.scope.arg_names:
            arg_idx = self.scope.arg_names.index(name)
            if self.translator.arg_needs_pointer(
                    self.scope.subprogram_name, arg_idx):
                return "*"+name
            else:
                return name
        elif shape not in [(), None]:
            return "*"+name
        else:
            return name

    def map_literal(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        if expr.dtype.kind == "c":
            r, i = expr.value
            return "%s_new(%s, %s)" % (
                    complex_type_name(expr.dtype),
                    self.rec(r, PREC_NONE),
                    self.rec(i, PREC_NONE))
        else:
            return expr.value

    def map_wildcard(self, expr, enclosing_prec):
        return ":"


# }}}

class Scope(object):
    def __init__(self, subprogram_name, arg_names=set()):
        self.subprogram_name = subprogram_name

        # map name to data
        self.data_statements = {}

        # map first letter to type
        self.implicit_types = {}

        # map name to dim tuple
        self.dim_map = {}

        # map name to dim tuple
        self.type_map = {}

        # map name to data
        self.data = {}

        self.arg_names = arg_names

        self.used_names = set()

        self.type_inf_mapper = None

    def known_names(self):
        return (self.used_names
                | set(six.iterkeys(self.dim_map))
                | set(six.iterkeys(self.type_map)))

    def is_known(self, name):
        return (name in self.used_names
                or name in self.dim_map
                or name in self.type_map)

    def use_name(self, name):
        self.used_names.add(name)

    def get_type(self, name):
        try:
            return self.type_map[name]
        except KeyError:

            if self.implicit_types is None:
                raise TranslationError(
                        "no type for '%s' found in implict none routine"
                        % name)

            return self.implicit_types.get(name[0], np.dtype(np.int32))

    def get_shape(self, name):
        return self.dim_map.get(name, ())

    def get_type_inference_mapper(self):
        if self.type_inf_mapper is None:
            self.type_inf_mapper = TypeInferenceMapper(self)

        return self.type_inf_mapper

    def translate_var_name(self, name):
        shape = self.dim_map.get(name)
        if name in self.data and shape is not None:
            return "%s_%s" % (self.subprogram_name, name)
        else:
            return name


class FTreeWalkerBase(object):
    def __init__(self):
        self.scope_stack = []

        self.expr_parser = FortranExpressionParser(self)

    def rec(self, expr, *args, **kwargs):
        mro = list(type(expr).__mro__)
        dispatch_class = kwargs.pop("dispatch_class", type(self))

        while mro:
            method_name = "map_"+mro.pop(0).__name__

            try:
                method = getattr(dispatch_class, method_name)
            except AttributeError:
                pass
            else:
                return method(self, expr, *args, **kwargs)

        raise NotImplementedError(
                "%s does not know how to map type '%s'"
                % (type(self).__name__,
                    type(expr)))

    ENTITY_RE = re.compile(
            r"^(?P<name>[_0-9a-zA-Z]+)"
            "(\((?P<shape>[-+*0-9:a-zA-Z,]+)\))?$")

    def parse_dimension_specs(self, dim_decls):
        def parse_bounds(bounds_str):
            start_end = bounds_str.split(":")

            assert 1 <= len(start_end) <= 2

            return tuple(self.parse_expr(s) for s in start_end)

        for decl in dim_decls:
            entity_match = self.ENTITY_RE.match(decl)
            assert entity_match

            groups = entity_match.groupdict()
            name = groups["name"]
            assert name

            if groups["shape"]:
                shape = [parse_bounds(s) for s in groups["shape"].split(",")]
            else:
                shape = None

            yield name, shape

    def __call__(self, expr, *args, **kwargs):
        return self.rec(expr, *args, **kwargs)

    # {{{ expressions

    def parse_expr(self, expr_str):
        return self.expr_parser(expr_str)

    # }}}


class ArgumentAnalayzer(FTreeWalkerBase):
    def __init__(self):
        FTreeWalkerBase.__init__(self)

        # map (func, arg_nr) to
        # 'w' for 'needs pointer'
        # [] for no obstacle to de-pointerification known
        # [(func_name, arg_nr), ...] # depends on how this arg is used

        self.arg_usage_info = {}

    def arg_needs_pointer(self, func, arg_nr):
        data = self.arg_usage_info.get((func, arg_nr), [])

        if isinstance(data, list):
            return any(
                    self.arg_needs_pointer(sub_func, sub_arg_nr)
                    for sub_func, sub_arg_nr in data)

        return True

    # {{{ map_XXX functions

    def map_BeginSource(self, node):
        scope = Scope(None)
        self.scope_stack.append(scope)

        for c in node.content:
            self.rec(c)

    def map_Subroutine(self, node):
        scope = Scope(node.name, list(node.args))
        self.scope_stack.append(scope)

        for c in node.content:
            self.rec(c)

        self.scope_stack.pop()

    def map_EndSubroutine(self, node):
        pass

    def map_Implicit(self, node):
        pass

    # {{{ types, declarations

    def map_Equivalence(self, node):
        raise NotImplementedError("equivalence")

    def map_Dimension(self, node):
        scope = self.scope_stack[-1]

        for name, shape in self.parse_dimension_specs(node.items):
            if name in scope.arg_names:
                arg_idx = scope.arg_names.index(name)
                self.arg_usage_info[scope.subprogram_name, arg_idx] = "w"

    def map_External(self, node):
        pass

    def map_type_decl(self, node):
        scope = self.scope_stack[-1]

        for name, shape in self.parse_dimension_specs(node.entity_decls):
            if shape is not None and name in scope.arg_names:
                arg_idx = scope.arg_names.index(name)
                self.arg_usage_info[scope.subprogram_name, arg_idx] = "w"

    map_Logical = map_type_decl
    map_Integer = map_type_decl
    map_Real = map_type_decl
    map_Complex = map_type_decl

    # }}}

    def map_Data(self, node):
        pass

    def map_Parameter(self, node):
        raise NotImplementedError("parameter")

    # {{{ I/O

    def map_Open(self, node):
        pass

    def map_Format(self, node):
        pass

    def map_Write(self, node):
        pass

    def map_Print(self, node):
        pass

    def map_Read1(self, node):
        pass

    # }}}

    def map_Assignment(self, node):
        scope = self.scope_stack[-1]

        lhs = self.parse_expr(node.variable)

        if isinstance(lhs, p.Subscript):
            lhs_name = lhs.aggregate.name
        elif isinstance(lhs, p.Call):
            # in absence of dim info, subscripts get parsed as calls
            lhs_name = lhs.function.name
        else:
            lhs_name = lhs.name

        if lhs_name in scope.arg_names:
            arg_idx = scope.arg_names.index(lhs_name)
            self.arg_usage_info[scope.subprogram_name, arg_idx] = "w"

    def map_Allocate(self, node):
        raise NotImplementedError("allocate")

    def map_Deallocate(self, node):
        raise NotImplementedError("deallocate")

    def map_Save(self, node):
        raise NotImplementedError("save")

    def map_Line(self, node):
        raise NotImplementedError

    def map_Program(self, node):
        raise NotImplementedError

    def map_Entry(self, node):
        raise NotImplementedError

    # {{{ control flow

    def map_Goto(self, node):
        pass

    def map_Call(self, node):
        scope = self.scope_stack[-1]

        for i, arg_str in enumerate(node.items):
            arg = self.parse_expr(arg_str)
            if isinstance(arg, (p.Variable, p.Subscript)):
                if isinstance(arg, p.Subscript):
                    arg_name = arg.aggregate.name
                else:
                    arg_name = arg.name

                if arg_name in scope.arg_names:
                    arg_idx = scope.arg_names.index(arg_name)
                    arg_usage = self.arg_usage_info.setdefault(
                            (scope.subprogram_name, arg_idx),
                            [])
                    if isinstance(arg_usage, list):
                        arg_usage.append((node.designator, i))

    def map_Return(self, node):
        pass

    def map_ArithmeticIf(self, node):
        pass

    def map_If(self, node):
        for c in node.content:
            self.rec(c)

    def map_IfThen(self, node):
        for c in node.content:
            self.rec(c)

    def map_ElseIf(self, node):
        pass

    def map_Else(self, node):
        pass

    def map_EndIfThen(self, node):
        pass

    def map_Do(self, node):
        for c in node.content:
            self.rec(c)

    def map_EndDo(self, node):
        pass

    def map_Continue(self, node):
        pass

    def map_Stop(self, node):
        pass

    def map_Comment(self, node):
        pass

    # }}}

    # }}}


# {{{ translator

class F2CLTranslator(FTreeWalkerBase):
    def __init__(self, addr_space_hints, force_casts, arg_info,
            use_restrict_pointers):
        FTreeWalkerBase.__init__(self)
        self.addr_space_hints = addr_space_hints
        self.force_casts = force_casts
        self.arg_info = arg_info
        self.use_restrict_pointers = use_restrict_pointers

    def arg_needs_pointer(self, subprogram_name, arg_index):
        return self.arg_info.arg_needs_pointer(subprogram_name, arg_index)

    # {{{ declaration helpers

    def get_declarator(self, name):
        scope = self.scope_stack[-1]
        return POD(scope.get_type(name), name)

    def get_declarations(self):
        scope = self.scope_stack[-1]

        result = []
        pre_func_decl = []

        def gen_shape(start_end):
            return ":".join(self.gen_expr(s) for s in start_end)

        for name in sorted(scope.known_names()):
            shape = scope.dim_map.get(name)

            if shape is not None:
                dim_stmt = cgen.Statement(
                    "dimension \"fortran\" %s[%s]" % (
                        scope.translate_var_name(name),
                        ", ".join(gen_shape(s) for s in shape)
                        ))

                # cannot omit 'dimension' decl even for rank-1 args:
                result.append(dim_stmt)

            if name in scope.data:
                assert name not in scope.arg_names

                data = scope.data[name]

                if shape is None:
                    assert len(data) == 1
                    result.append(
                            cgen.Initializer(
                                self.get_declarator(name),
                                self.gen_expr(data[0])
                                ))
                else:
                    from cgen.opencl import CLConstant
                    pre_func_decl.append(
                            cgen.Initializer(
                                CLConstant(
                                    cgen.ArrayOf(self.get_declarator(
                                        "%s_%s" % (scope.subprogram_name, name)))),
                                "{ %s }" % ",\n".join(self.gen_expr(x) for x in data)
                                ))
            else:
                if name not in scope.arg_names:
                    if shape is not None:
                        result.append(cgen.Statement(
                            "%s %s[nitemsof(%s)]"
                            % (
                                dtype_to_ctype(scope.get_type(name)),
                                name, name)))
                    else:
                        result.append(self.get_declarator(name))

        return pre_func_decl, result

    def map_statement_list(self, content):
        body = []

        for c in content:
            mapped = self.rec(c)
            if mapped is None:
                warn("mapping '%s' returned None" % type(c))
            elif isinstance(mapped, list):
                body.extend(mapped)
            else:
                body.append(mapped)

        return body

    # }}}

    # {{{ map_XXX functions

    def map_BeginSource(self, node):
        scope = Scope(None)
        self.scope_stack.append(scope)

        return self.map_statement_list(node.content)

    def map_Subroutine(self, node):
        assert not node.prefix
        assert not hasattr(node, "suffix")

        scope = Scope(node.name, list(node.args))
        self.scope_stack.append(scope)

        body = self.map_statement_list(node.content)

        pre_func_decl, in_func_decl = self.get_declarations()
        body = in_func_decl + [cgen.Line()] + body

        if isinstance(body[-1], cgen.Statement) and body[-1].text == "return":
            body.pop()

        def get_arg_decl(arg_idx, arg_name):
            decl = self.get_declarator(arg_name)

            if self.arg_needs_pointer(node.name, arg_idx):
                hint = self.addr_space_hints.get((node.name, arg_name))
                if hint:
                    decl = hint(cgen.Pointer(decl))
                else:
                    if self.use_restrict_pointers:
                        decl = cgen.RestrictPointer(decl)
                    else:
                        decl = cgen.Pointer(decl)

            return decl

        result = cgen.FunctionBody(
                cgen.FunctionDeclaration(
                    cgen.Value("void", node.name),
                    [get_arg_decl(i, arg) for i, arg in enumerate(node.args)]
                    ),
                cgen.Block(body))

        self.scope_stack.pop()
        if pre_func_decl:
            return pre_func_decl + [cgen.Line(), result]
        else:
            return result

    def map_EndSubroutine(self, node):
        return []

    def map_Implicit(self, node):
        scope = self.scope_stack[-1]

        if not node.items:
            assert not scope.implicit_types
            scope.implicit_types = None

        for stmt, specs in node.items:
            tp = self.dtype_from_stmt(stmt)
            for start, end in specs:
                for char_code in range(ord(start), ord(end)+1):
                    scope.implicit_types[chr(char_code)] = tp

        return []

    # {{{ types, declarations

    def map_Equivalence(self, node):
        raise NotImplementedError("equivalence")

    TYPE_MAP = {
            ("real", "4"): np.float32,
            ("real", "8"): np.float64,
            ("real", "16"): np.float128,

            ("complex", "8"): np.complex64,
            ("complex", "16"): np.complex128,
            ("complex", "32"): np.complex256,

            ("integer", ""): np.int32,
            ("integer", "4"): np.int32,
            ("complex", "8"): np.int64,
            }

    def dtype_from_stmt(self, stmt):
        length, kind = stmt.selector
        assert not kind
        return np.dtype(self.TYPE_MAP[(type(stmt).__name__.lower(), length)])

    def map_type_decl(self, node):
        scope = self.scope_stack[-1]

        tp = self.dtype_from_stmt(node)

        for name, shape in self.parse_dimension_specs(node.entity_decls):
            if shape is not None:
                assert name not in scope.dim_map
                scope.dim_map[name] = shape
                scope.use_name(name)

            assert name not in scope.type_map
            scope.type_map[name] = tp

        return []

    map_Logical = map_type_decl
    map_Integer = map_type_decl
    map_Real = map_type_decl
    map_Complex = map_type_decl

    def map_Dimension(self, node):
        scope = self.scope_stack[-1]

        for name, shape in self.parse_dimension_specs(node.items):
            if shape is not None:
                assert name not in scope.dim_map
                scope.dim_map[name] = shape
                scope.use_name(name)

        return []

    def map_External(self, node):
        raise NotImplementedError("external")

    # }}}

    def map_Data(self, node):
        scope = self.scope_stack[-1]

        for name, data in node.stmts:
            name, = name
            assert name not in scope.data
            scope.data[name] = [self.parse_expr(i) for i in data]

        return []

    def map_Parameter(self, node):
        raise NotImplementedError("parameter")

    # {{{ I/O

    def map_Open(self, node):
        raise NotImplementedError

    def map_Format(self, node):
        warn("'format' unsupported", TranslatorWarning)

    def map_Write(self, node):
        warn("'write' unsupported", TranslatorWarning)

    def map_Print(self, node):
        warn("'print' unsupported", TranslatorWarning)

    def map_Read1(self, node):
        warn("'read' unsupported", TranslatorWarning)

    # }}}

    def map_Assignment(self, node):
        lhs = self.parse_expr(node.variable)
        from pymbolic.primitives import Subscript
        if isinstance(lhs, Subscript):
            lhs_name = lhs.aggregate.name
        else:
            lhs_name = lhs.name

        scope = self.scope_stack[-1]
        scope.use_name(lhs_name)
        infer_type = scope.get_type_inference_mapper()

        rhs = self.parse_expr(node.expr)
        lhs_dtype = infer_type(lhs)
        rhs_dtype = infer_type(rhs)

        # check for silent truncation of complex
        if lhs_dtype.kind != 'c' and rhs_dtype.kind == 'c':
            from pymbolic import var
            rhs = var("real")(rhs)
        # check for silent widening of real
        if lhs_dtype.kind == 'c' and rhs_dtype.kind != 'c':
            from pymbolic import var
            rhs = var("fromreal")(rhs)

        return cgen.Assign(self.gen_expr(lhs), self.gen_expr(rhs))

    def map_Allocate(self, node):
        raise NotImplementedError("allocate")

    def map_Deallocate(self, node):
        raise NotImplementedError("deallocate")

    def map_Save(self, node):
        raise NotImplementedError("save")

    def map_Line(self, node):
        #from warnings import warn
        #warn("Encountered a 'line': %s" % node)
        raise NotImplementedError

    def map_Program(self, node):
        raise NotImplementedError

    def map_Entry(self, node):
        raise NotImplementedError

    # {{{ control flow

    def map_Goto(self, node):
        return cgen.Statement("goto label_%s" % node.label)

    def map_Call(self, node):
        def transform_arg(i, arg_str):
            expr = self.parse_expr(arg_str)
            result = self.gen_expr(expr)
            if self.arg_needs_pointer(node.designator, i):
                result = "&"+result

            cast = self.force_casts.get(
                    (node.designator, i))
            if cast is not None:
                result = "(%s) (%s)" % (cast, result)

            return result

        return cgen.Statement("%s(%s)" % (
            node.designator,
            ", ".join(transform_arg(i, arg_str)
                for i, arg_str in enumerate(node.items))))

    def map_Return(self, node):
        return cgen.Statement("return")

    def map_ArithmeticIf(self, node):
        raise NotImplementedError

    def map_If(self, node):
        return cgen.If(self.transform_expr(node.expr),
                self.rec(node.content[0]))

    def map_IfThen(self, node):
        current_cond = self.transform_expr(node.expr)

        blocks_and_conds = []
        else_block = []

        def end_block():
            if current_body:
                if current_cond is None:
                    else_block[:] = self.map_statement_list(current_body)
                else:
                    blocks_and_conds.append(
                            (current_cond, cgen.block_if_necessary(
                                self.map_statement_list(current_body))))

            del current_body[:]

        from fparser.statements import Else, ElseIf
        i = 0
        current_body = []
        while i < len(node.content):
            c = node.content[i]
            if isinstance(c, ElseIf):
                end_block()
                current_cond = self.transform_expr(c.expr)
            elif isinstance(c, Else):
                end_block()
                current_cond = None
            else:
                current_body.append(c)

            i += 1
        end_block()

        def block_or_none(body):
            if not body:
                return None
            else:
                return cgen.block_if_necessary(body)

        return cgen.make_multiple_ifs(
                blocks_and_conds,
                block_or_none(else_block))

    def map_EndIfThen(self, node):
        return []

    def map_Do(self, node):
        scope = self.scope_stack[-1]

        body = self.map_statement_list(node.content)

        if node.loopcontrol:
            loop_var, loop_bounds = node.loopcontrol.split("=")
            loop_var = loop_var.strip()
            scope.use_name(loop_var)
            loop_bounds = [self.parse_expr(s) for s in loop_bounds.split(",")]

            if len(loop_bounds) == 2:
                start, stop = loop_bounds
                step = 1
            elif len(loop_bounds) == 3:
                start, stop, step = loop_bounds
            else:
                raise RuntimeError("loop bounds not understood: %s"
                        % node.loopcontrol)

            if not isinstance(step, int):
                print(type(step))
                raise TranslationError(
                        "non-constant steps not yet supported: %s" % step)

            if step < 0:
                comp_op = ">="
            else:
                comp_op = "<="

            return cgen.For(
                    "%s = %s" % (loop_var, self.gen_expr(start)),
                    "%s %s %s" % (loop_var, comp_op, self.gen_expr(stop)),
                    "%s += %s" % (loop_var, self.gen_expr(step)),
                    cgen.block_if_necessary(body))

        else:
            raise NotImplementedError("unbounded do loop")

    def map_EndDo(self, node):
        return []

    def map_Continue(self, node):
        return cgen.Statement("label_%s:" % node.label)

    def map_Stop(self, node):
        raise NotImplementedError("stop")

    def map_Comment(self, node):
        if node.content:
            return cgen.LineComment(node.content.strip())
        else:
            return []

    # }}}

    # }}}

    # {{{ expressions

    def gen_expr(self, expr):
        scope = self.scope_stack[-1]
        return CCodeMapper(self, scope)(expr)

    def transform_expr(self, expr_str):
        return self.gen_expr(self.expr_parser(expr_str))

    # }}}

# }}}


def f2cl(source, free_form=False, strict=True,
        addr_space_hints={}, force_casts={},
        do_arg_analysis=True,
        use_restrict_pointers=False,
        try_compile=False):
    from fparser import api
    tree = api.parse(source, isfree=free_form, isstrict=strict,
            analyze=False, ignore_comments=False)

    arg_info = ArgumentAnalayzer()
    if do_arg_analysis:
        arg_info(tree)

    source = F2CLTranslator(addr_space_hints, force_casts,
            arg_info, use_restrict_pointers=use_restrict_pointers)(tree)

    func_decls = []
    for entry in source:
        if isinstance(entry, cgen.FunctionBody):
            func_decls.append(entry.fdecl)

    mod = cgen.Module(func_decls + [cgen.Line()] + source)

    #open("pre-cnd.cl", "w").write(str(mod))

    from cnd import transform_cl
    str_mod = transform_cl(str(mod))

    if try_compile:
        import pyopencl as cl
        ctx = cl.create_some_context()
        cl.Program(ctx, """
            #if __OPENCL_VERSION__ <= CL_VERSION_1_1
            #pragma OPENCL EXTENSION cl_khr_fp64: enable
            #endif
            #include <pyopencl-complex.h>
            """).build()
    return str_mod


def f2cl_files(source_file, target_file, **kwargs):
    mod = f2cl(open(source_file).read(), **kwargs)
    open(target_file, "w").write(mod)


if __name__ == "__main__":
    import logging
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('fparser').addHandler(console)

    from cgen.opencl import CLConstant

    if 0:
        f2cl_files("hank107.f", "hank107.cl",
                addr_space_hints={
                    ("hank107p", "p"): CLConstant,
                    ("hank107pc", "p"): CLConstant,
                    },
                force_casts={
                    ("hank107p", 0): "__constant cdouble_t *",
                    })

        f2cl_files("cdjseval2d.f", "cdjseval2d.cl")

    f2cl_files("hank103.f", "hank103.cl",
            addr_space_hints={
                ("hank103p", "p"): CLConstant,
                ("hank103pc", "p"): CLConstant,
                },
            force_casts={
                ("hank103p", 0): "__constant cdouble_t *",
                },
            try_compile=True)

# vim: foldmethod=marker
