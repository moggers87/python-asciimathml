# Copyright (c) 2010-2011, Gabriele Favalessa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from itertools import chain
from xml.etree.ElementTree import Element, tostring
import re
import sys

__all__ = ['parse']

Element_ = Element
AtomicString_ = lambda s: s

DELIMITERS = {'{': '}', '(': ')', '[': ']'}
NUMBER_RE = re.compile(r'-?(\d+\.(\d+)?|\.?\d+)')
QUOTED_STRING_RE = re.compile(r'"([^"]*)"')


def text_check(text):
    py2str = (sys.version_info.major == 2 and isinstance(text, basestring))
    py3str = (sys.version_info.major >= 3 and isinstance(text, str))

    return (py3str or py2str)


def element_factory(tag, text=None, *children, **attrib):
    element = Element_(tag, **attrib)

    if not text is None:
        if text_check(text):
            element.text = AtomicString_(text)
        else:
            children = (text, ) + children

    for child in children:
        element.append(child)

    return element


def strip_parens(n):
    if n.tag == 'mrow':
        if n[0].get('_opening', False):
            del n[0]

        if n[-1].get('_closing', False):
            del n[-1]

    return n


def strip_tags(n):
    return ''.join(e.text for e in n)


def is_enclosed_in_parens(n):
    return n.tag == 'mrow' and n[0].get('_opening', False) and n[-1].get('_closing', False)


def binary(operator, operand_1, operand_2, swap=False, o1_attr=None, o2_attr=None):
    operand_1 = strip_parens(operand_1)
    operand_2 = strip_parens(operand_2)

    if swap:
        operand_1, operand_2 = operand_2, operand_1

    if o1_attr is None:
        operator.append(operand_1)
    else:
        operator.attrib[o1_attr] = strip_tags(operand_1)

    if o2_attr is None:
        operator.append(operand_2)
    else:
        operator.attrib[o2_attr] = strip_tags(operand_2)

    return operator


def unary(operator, operand, swap=False, rewrite_lr=None):
    operand = strip_parens(operand)

    if rewrite_lr is None:
        rewrite_lr = []

    if rewrite_lr:
        opener = element_factory("mo", rewrite_lr[0], _opening=True)
        closer = element_factory("mo", rewrite_lr[1], _closing=True)

        if is_enclosed_in_parens(operand):
            operand[0] = opener
            operand[-1] = closer
            return operand
        else:
            operator.append(opener)
            operator.append(operand)
            operator.append(closer)
    else:
        if swap:
            operator.insert(0, operand)
        else:
            operator.append(operand)

    return operator


def frac(num, den):
    return element_factory('mfrac', strip_parens(num), strip_parens(den))


def sub(base, subscript):
    subscript = strip_parens(subscript)

    if base.tag in ('msup', 'mover'):
        children = base.getchildren()
        n = element_factory(
            'msubsup' if base.tag == 'msup' else 'munderover',
            children[0], subscript, children[1]
        )
    else:
        n = element_factory(
            'munder' if base.get('_underover', False) else 'msub',
            base, subscript
        )

    return n


def sup(base, superscript):
    superscript = strip_parens(superscript)

    if base.tag in ('msub', 'munder'):
        children = base.getchildren()
        n = element_factory(
            'msubsup' if base.tag == 'msub' else 'munderover',
            children[0], children[1], superscript
        )
    else:
        n = element_factory(
            'mover' if base.get('_underover', False) else 'msup',
            base, superscript
        )

    return n


def parse(s, element=Element, atomicstring=lambda s: s):
    """
    Translates from ASCIIMathML (an easy to type and highly readable way to
    represent math formulas) into MathML (a w3c standard directly displayable by
    some web browsers).

    The function `parse()` generates a tree of elements:

    >>> import asciimathml
    >>> asciimathml.parse('sqrt 2')
    <Element math at b76fb28c>

    The tree can then be manipulated using the standard python library.  For
    example we can generate its string representation:

    >>> from xml.etree.ElementTree import tostring
    >>> tostring(asciimathml.parse('sqrt 2'))
    '<math><mstyle><msqrt><mn>2</mn></msqrt></mstyle></math>'
    """

    global Element_, AtomicString_

    Element_ = element
    AtomicString_ = atomicstring

    s, nodes = parse_exprs(s)
    remove_invisible(nodes)
    nodes = map(remove_private, nodes)

    return element_factory('math', element_factory('mstyle', *nodes))


def parse_string(s):
    opening = s[0]

    if opening in DELIMITERS:
        closing = DELIMITERS[opening]
        end = s.find(closing)

        text = s[1:end]
        s = s[end+1:]

        children = []
        if text.startswith(' '):
            children.append(element_factory('mspace', width='1ex'))
        children.append(element_factory('mtext', text))
        if text.endswith(' '):
            children.append(element_factory('mspace', width='1ex'))

        return s, element_factory('mrow', *children)
    else:
        s, text = parse_m(s)
        return s, element_factory('mtext', text)


def trace_parser(p):
    """
    Decorator for tracing the parser.

    Use it to decorate functions with signature:

      string -> (string, nodes)

    and a trace of the progress made by the parser will be printed to stderr.

    Currently parse_exprs(), parse_expr() and parse_m() have the right signature.
    """
    tracing_level = 0

    def nodes_to_string(n):
        if isinstance(n, list):
            result = '[ '
            for m in map(nodes_to_string, n):
                result += m
                result += ' '
            result += ']'

            return result
        else:
            try:
                return tostring(remove_private(copy(n)))
            except Exception as e:
                return n

    def print_trace(*args):
        sys.stderr.write("    " * tracing_level)
        for arg in args:
            sys.stderr.write(str(arg))
            sys.stderr.write(' ')
        sys.stderr.write('\n')
        sys.stderr.flush()

    def wrapped(s, *args, **kwargs):
        nonlocal tracing_level

        print_trace(p.__name__, repr(s))

        tracing_level += 1
        s, n = p(s, *args, **kwargs)
        tracing_level -= 1

        print_trace("-> ", repr(s), nodes_to_string(n))

        return s, n

    return wrapped


def parse_expr(s, siblings, required=False):
    s, n = parse_m(s, required=required)

    if not n is None:
        # Being both an _opening and a _closing element is a trait of
        # symmetrical delimiters (e.g. ||).
        # In that case, act as an opening delimiter only if there is not
        # already one of the same kind among the preceding siblings.

        sym_paren = n.get('_opening', False) and n.get('_closing', False)
        prev_sib_pos = find_node_backwards(siblings, n.text)
        parens_nest = (
            n.get('_opening', False)
            and (
                not n.get('_closing', False)
                or (
                    sym_paren is (prev_sib_pos != -1)
                )
            )
        )

        if parens_nest:
            if sym_paren:
                n = element_factory('mrow', *chain(siblings[prev_sib_pos:], [n]))
                del siblings[prev_sib_pos:]
            else:
                s, children = parse_exprs(s, [n], inside_parens=True)
                n = element_factory('mrow', *children)

        if n.tag == 'mtext':
            s, n = parse_string(s)
        elif n.get('_arity', 0) == 1:
            s, m = parse_expr(s, [], True)
            n = unary(n, m, swap=n.get('_swap', False), rewrite_lr=n.get('_rewrite_lr', []))
        elif n.get('_arity', 0) == 2:
            s, m1 = parse_expr(s, [], True)
            s, m2 = parse_expr(s, [], True)
            n = binary(
                n, m1, m2,
                swap=n.get('_swap', False),
                o1_attr=n.get('_o1_attr', None),
                o2_attr=n.get('_o2_attr', None)
            )

    return s, n


def find_node(ns, text):
    for i, n in enumerate(ns):
        if n.text == text:
            return i

    return -1


def find_node_backwards(ns, text):
    for i, n in enumerate(reversed(ns)):
        if n.text == text:
            return len(ns) - (i + 1)

    return -1


def nodes_to_row(row):
    mrow = element_factory('mtr')

    nodes = row.getchildren()

    while True:
        i = find_node(nodes, ',')

        if i > 0:
            mrow.append(element_factory('mtd', *nodes[:i]))

            nodes = nodes[i+1:]
        else:
            mrow.append(element_factory('mtd', *nodes))
            break

    return mrow


def nodes_to_matrix(nodes):
    mtable = element_factory('mtable')

    for row in nodes[1:-1]:
        if row.text == ',':
            continue

        mtable.append(nodes_to_row(strip_parens(row)))

    return element_factory('mrow', nodes[0], mtable, nodes[-1])


def parse_exprs(s, nodes=None, inside_parens=False):
    if nodes is None:
        nodes = []

    inside_matrix = False

    while True:
        s, n = parse_expr(s, nodes)

        if not n is None:
            truly_closing = (
                n.get('_closing', False)
                and (
                    not n.get('_opening', False)
                    or
                    (
                        find_node_backwards(nodes, n.text) != -1
                    )
                )
            )

            neg_number = (
                n.tag == 'mrow'
                and len(n) == 2
                and n[0].text == '-'
                and n[1].tag in {'mn', 'mi'}
            )
            term_before = (
                nodes
                and nodes[-1].tag != 'mo'
            )

            if neg_number and term_before:
                nodes.extend(n)
            else:
                nodes.append(n)

            if truly_closing:
                if not inside_matrix:
                    return s, nodes
                else:
                    return s, nodes_to_matrix(nodes)

            if inside_parens and n.text == ',' and is_enclosed_in_parens(nodes[-2]):
                inside_matrix = True

            len_nodes = len(nodes)
            if len_nodes >= 3 and nodes[-2].get('_special_binary'):
                transform = nodes[-2].get('_special_binary')
                nodes[-3:] = [transform(nodes[-3], nodes[-1])]
            elif s == '' and len_nodes == 2 and nodes[-1].get('_special_binary'):
                transform = nodes[-1].get('_special_binary')
                nodes[-2:] = [transform(nodes[-2], element_factory("mo"))]

        if s == '':
            return '', nodes


def remove_private(n):
    _ks = [k for k in n.keys() if k.startswith('_') or k == 'attrib']

    for _k in _ks:
        del n.attrib[_k]

    for c in n.getchildren():
        remove_private(c)

    return n


def remove_invisible(ns, parent=None):
    for i in range(len(ns)-1, -1, -1):
        if ns[i].get('_invisible', False):
            if parent is None:
                del ns[i]
            else:
                parent.remove(ns[i])
        else:
            remove_invisible(ns[i].getchildren(), parent=ns[i])


def copy(n):
    m = element_factory(n.tag, n.text, **dict(n.items()))

    for c in n.getchildren():
        m.append(copy(c))

    return m


def parse_m(s, required=False):
    s = s.strip()

    if s == '':
        return '', element_factory('mi', u'\u25a1') if required else None

    m = QUOTED_STRING_RE.match(s)
    if m:
        text = m.group(1)

        children = []
        if text.startswith(' '):
            children.append(element_factory('mspace', width='1ex'))
        children.append(element_factory('mtext', text))
        if text.endswith(' '):
            children.append(element_factory('mspace', width='1ex'))

        return s[m.end():], element_factory(
            'mrow',
            *children
        )

    m = NUMBER_RE.match(s)

    if m:
        number = m.group(0)
        if number[0] == '-':
            return s[m.end():], element_factory(
                'mrow',
                element_factory('mo', '-'),
                element_factory('mn', number[1:])
            )
        else:
            return s[m.end():], element_factory('mn', number)

    for y in symbol_names:
        if s.startswith(y):
            n = copy(symbols[y])

            if n.get('_space', False):
                n = element_factory(
                    'mrow',
                    element_factory('mspace', width='1ex'),
                    n,
                    element_factory('mspace', width='1ex')
                )

            return s[len(y):], n

    return s[1:], element_factory('mi' if s[0].isalpha() else 'mo', s[0])


symbols = {
    "alpha": element_factory("mi", u"\u03B1"),,
    "beta": element_factory("mi", u"\u03B2"),
    "chi": element_factory("mi", u"\u03C7"),
    "delta": element_factory("mi", u"\u03B4"),
    "Delta": element_factory("mo", u"\u0394"),
    "epsi": element_factory("mi", u"\u03B5"),
    "varepsilon": element_factory("mi", u"\u025B"),
    "eta": element_factory("mi", u"\u03B7"),
    "gamma": element_factory("mi", u"\u03B3"),
    "Gamma": element_factory("mo", u"\u0393"),
    "iota": element_factory("mi", u"\u03B9"),
    "kappa": element_factory("mi", u"\u03BA"),
    "lambda": element_factory("mi", u"\u03BB"),
    "Lambda": element_factory("mo", u"\u039B"),
    "lamda": element_factory("mi", u"\u03BB"),
    "Lamda": element_factory("mo", u"\u039B"),
    "mu": element_factory("mi", u"\u03BC"),
    "nu": element_factory("mi", u"\u03BD"),
    "omega": element_factory("mi", u"\u03C9"),
    "Omega": element_factory("mo", u"\u03A9"),
    "phi": element_factory("mi", u"\u03C6"),
    "varphi": element_factory("mi", u"\u03D5"),
    "Phi": element_factory("mo", u"\u03A6"),
    "pi": element_factory("mi", u"\u03C0"),
    "Pi": element_factory("mo", u"\u03A0"),
    "psi": element_factory("mi", u"\u03C8"),
    "Psi": element_factory("mi", u"\u03A8"),
    "rho": element_factory("mi", u"\u03C1"),
    "sigma": element_factory("mi", u"\u03C3"),
    "Sigma": element_factory("mo", u"\u03A3"),
    "tau": element_factory("mi", u"\u03C4"),
    "theta": element_factory("mi", u"\u03B8"),
    "vartheta": element_factory("mi", u"\u03D1"),
    "Theta": element_factory("mo", u"\u0398"),
    "upsilon": element_factory("mi", u"\u03C5"),
    "xi": element_factory("mi", u"\u03BE"),
    "Xi": element_factory("mo", u"\u039E"),
    "zeta": element_factory("mi", u"\u03B6"),

    "*": element_factory("mo", u"\u22C5"),
    "**": element_factory("mo", u"\u2217"),
    "***": element_factory("mo", u"\u22C6"),

    "/": element_factory("mo", u"/", _special_binary=frac),
    "^": element_factory("mo", u"^", _special_binary=sup),
    "_": element_factory("mo", u"_", _special_binary=sub),
    "//": element_factory("mo", u"/"),
    "\\\\": element_factory("mo", u"\\"),
    "setminus": element_factory("mo", u"\\"),
    "xx": element_factory("mo", u"\u00D7"),
    "|><": element_factory("mo", u"\u22C9"),
    "><|": element_factory("mo", u"\u22CA"),
    "|><|": element_factory("mo", u"\u22C8"),
    "-:": element_factory("mo", u"\u00F7"),
    "@": element_factory("mo", u"\u2218"),
    "o+": element_factory("mo", u"\u2295"),
    "ox": element_factory("mo", u"\u2297"),
    "o.": element_factory("mo", u"\u2299"),
    "sum": element_factory("mo", u"\u2211", _underover=True),
    "prod": element_factory("mo", u"\u220F", _underover=True),
    "^^": element_factory("mo", u"\u2227"),
    "^^^": element_factory("mo", u"\u22C0", _underover=True),
    "vv": element_factory("mo", u"\u2228"),
    "vvv": element_factory("mo", u"\u22C1", _underover=True),
    "nn": element_factory("mo", u"\u2229"),
    "nnn": element_factory("mo", u"\u22C2", _underover=True),
    "uu": element_factory("mo", u"\u222A"),
    "uuu": element_factory("mo", u"\u22C3", _underover=True),

    "!=": element_factory("mo", u"\u2260"),
    ":=": element_factory("mo", u":="),
    "lt": element_factory("mo", u"<"),
    "gt": element_factory("mo", u">"),
    "<=": element_factory("mo", u"\u2264"),
    "lt=": element_factory("mo", u"\u2264"),
    "gt=": element_factory("mo", u"\u2265"),
    ">=": element_factory("mo", u"\u2265"),
    "geq": element_factory("mo", u"\u2265"),
    "-<": element_factory("mo", u"\u227A"),
    "-lt": element_factory("mo", u"\u227A"),
    ">-": element_factory("mo", u"\u227B"),
    "-<=": element_factory("mo", u"\u2AAF"),
    ">-=": element_factory("mo", u"\u2AB0"),
    "in": element_factory("mo", u"\u2208"),
    "!in": element_factory("mo", u"\u2209"),
    "sub": element_factory("mo", u"\u2282"),
    "sup": element_factory("mo", u"\u2283"),
    "sube": element_factory("mo", u"\u2286"),
    "supe": element_factory("mo", u"\u2287"),
    "-=": element_factory("mo", u"\u2261"),
    "~=": element_factory("mo", u"\u2245"),
    "~~": element_factory("mo", u"\u2248"),
    "prop": element_factory("mo", u"\u221D"),

    "and": element_factory("mtext", u"and", _space=True),
    "or": element_factory("mtext", u"or", _space=True),
    "not": element_factory("mo", u"\u00AC"),
    "=>": element_factory("mo", u"\u21D2"),
    "if": element_factory("mo", u"if", _space=True),
    "<=>": element_factory("mo", u"\u21D4"),
    "AA": element_factory("mo", u"\u2200"),
    "EE": element_factory("mo", u"\u2203"),
    "_|_": element_factory("mo", u"\u22A5"),
    "TT": element_factory("mo", u"\u22A4"),
    "|--": element_factory("mo", u"\u22A2"),
    "|==": element_factory("mo", u"\u22A8"),

    "(": element_factory("mo", "(", _opening=True),
    ")": element_factory("mo", ")", _closing=True),
    "[": element_factory("mo", "[", _opening=True),
    "]": element_factory("mo", "]", _closing=True),
    "{": element_factory("mo", "{", _opening=True),
    "}": element_factory("mo", "}", _closing=True),
    "|": element_factory("mo", u"|", _opening=True, _closing=True),
# double vertical line
    "||": element_factory("mo", u"\u2016", _opening=True, _closing=True),
    "(:": element_factory("mo", u"\u2329", _opening=True),
    ":)": element_factory("mo", u"\u232A", _closing=True),
    "<<": element_factory("mo", u"\u2329", _opening=True),
    ">>": element_factory("mo", u"\u232A", _closing=True),
    "{:": element_factory("mo", u"{:", _opening=True, _invisible=True),
    ":}": element_factory("mo", u":}", _closing=True, _invisible=True),

    "int": element_factory("mo", u"\u222B"),
#     "dx": element_factory("mi", u"{:d x:}", _definition=True),
#     "dy": element_factory("mi", u"{:d y:}", _definition=True),
#     "dz": element_factory("mi", u"{:d z:}", _definition=True),
#     "dt": element_factory("mi", u"{:d t:}", _definition=True),
    "oint": element_factory("mo", u"\u222E"),
    "del": element_factory("mo", u"\u2202"),
    "grad": element_factory("mo", u"\u2207"),
    "+-": element_factory("mo", u"\u00B1"),
    "O/": element_factory("mo", u"\u2205"),
    "oo": element_factory("mo", u"\u221E"),
    "aleph": element_factory("mo", u"\u2135"),
    "...": element_factory("mo", u"..."),
    ":.": element_factory("mo", u"\u2234"),
    "/_": element_factory("mo", u"\u2220"),
    "/_\\": element_factory("mo", u"\u25B3"),
    "'": element_factory("mo", u"\u2032"),
# arity of 1
    "tilde": element_factory("mover", element_factory("mo", u"~"), _arity=1, _swap=True),
    "\\ ": element_factory("mo", u"\u00A0"),
    "frown": element_factory("mo", u"\u2322"),
    "quad": element_factory("mo", u"\u00A0\u00A0"),
    "qquad": element_factory("mo", u"\u00A0\u00A0\u00A0\u00A0"),
    "cdots": element_factory("mo", u"\u22EF"),
    "vdots": element_factory("mo", u"\u22EE"),
    "ddots": element_factory("mo", u"\u22F1"),
    "diamond": element_factory("mo", u"\u22C4"),
    "square": element_factory("mo", u"\u25A1"),
    "|__": element_factory("mo", u"\u230A"),
    "__|": element_factory("mo", u"\u230B"),
    "|~": element_factory("mo", u"\u2308"),
    "~|": element_factory("mo", u"\u2309"),
    "CC": element_factory("mo", u"\u2102"),
    "NN": element_factory("mo", u"\u2115"),
    "QQ": element_factory("mo", u"\u211A"),
    "RR": element_factory("mo", u"\u211D"),
    "ZZ": element_factory("mo", u"\u2124"),
    "f": element_factory("mi", u"f", _func=True) # sample
    "g": element_factory("mi", u"g", _func=True),

    "lim": element_factory("mo", u"lim", _underover=True),
    "Lim": element_factory("mo", u"Lim", _underover=True),
    "sin": element_factory("mrow", element_factory("mo", "sin"), _arity=1),
    "sin": element_factory("mrow", element_factory("mo", "sin"), _arity=1),
    "cos": element_factory("mrow", element_factory("mo", "cos"), _arity=1),
    "tan": element_factory("mrow", element_factory("mo", "tan"), _arity=1),
    "sinh": element_factory("mrow", element_factory("mo", "sinh"), _arity=1),
    "cosh": element_factory("mrow", element_factory("mo", "cosh"), _arity=1),
    "tanh": element_factory("mrow", element_factory("mo", "tanh"), _arity=1),
    "cot": element_factory("mrow", element_factory("mo", "cot"), _arity=1),
    "sec": element_factory("mrow", element_factory("mo", "sec"), _arity=1),
    "csc": element_factory("mrow", element_factory("mo", "csc"), _arity=1),
    "log": element_factory("mrow", element_factory("mo", "log"), _arity=1),
    "arcsin": element_factory("mrow", element_factory("mo", "arcsin"), _arity=1),
    "arccos": element_factory("mrow", element_factory("mo", "arccos"), _arity=1),
    "arctan": element_factory("mrow", element_factory("mo", "arctan"), _arity=1),
    "coth": element_factory("mrow", element_factory("mo", "coth"), _arity=1),
    "sech": element_factory("mrow", element_factory("mo", "sech"), _arity=1),
    "csch": element_factory("mrow", element_factory("mo", "csch"), _arity=1),
    "exp": element_factory("mrow", element_factory("mo", "exp"), _arity=1),

    "abs": element_factory("mrow", element_factory("mo", "abs", _invisible=True), _arity=1, _rewrite_lr=[u"|", u"|"]),
    "norm": element_factory("mrow", element_factory("mo", "norm", _invisible=True), _arity=1, _rewrite_lr=[u"\u2225", u"\u2225"]),
    "floor": element_factory("mrow", element_factory("mo", "floor", _invisible=True), _arity=1, _rewrite_lr=[u"\u230A", u"\u230B"]),
    "ceil": element_factory("mrow", element_factory("mo", "ceil", _invisible=True), _arity=1, _rewrite_lr=[u"\u2308", u"\u2309"]),

    "ln": element_factory("mrow", element_factory("mo", "ln"), _arity=1),
    "det": element_factory("mrow", element_factory("mo", "det"), _arity=1),
    "gcd": element_factory("mrow", element_factory("mo", "gcd"), _arity=1),
    "lcm": element_factory("mrow", element_factory("mo", "lcm"), _arity=1),
    "dim": element_factory("mo", u"dim"),
    "mod": element_factory("mo", u"mod"),
    "lub": element_factory("mo", u"lub"),
    "glb": element_factory("mo", u"glb"),
    "min": element_factory("mo", u"min", _underover=True),
    "max": element_factory("mo", u"max", _underover=True),

    "uarr": element_factory("mo", u"\u2191"),
    "darr": element_factory("mo", u"\u2193"),
    "rarr": element_factory("mo", u"\u2192"),
    "->": element_factory("mo", u"\u2192"),
    "|->": element_factory("mo", u"\u21A6"),
    "larr": element_factory("mo", u"\u2190"),
    "harr": element_factory("mo", u"\u2194"),
    "rArr": element_factory("mo", u"\u21D2"),
    "lArr": element_factory("mo", u"\u21D0"),
    "hArr": element_factory("mo", u"\u21D4"),

    "hat": element_factory("mover", element_factory("mo", u"\u005E"), _arity=1, _swap=1),
    "bar": element_factory("mover", element_factory("mo", u"\u00AF"), _arity=1, _swap=1),
    "vec": element_factory("mover", element_factory("mo", u"\u2192"), _arity=1, _swap=1),
    "dot": element_factory("mover", element_factory("mo", u"."), _arity=1, _swap=1),
    "ddot": element_factory("mover", element_factory("mo", u".."), _arity=1, _swap=1),
    "ul": element_factory("munder", element_factory("mo", u"\u0332"), _arity=1, _swap=1),
    "ubrace": element_factory("munder", element_factory("mo",  u"\u23DF"), _swap=True, _arity=1),
    "obrace": element_factory("mover", element_factory("mo", u"\u23DE"), _swap=True, _arity=1),

    "sqrt": element_factory("msqrt", _arity=1),
    "root": element_factory("mroot", _arity=2, _swap=True),
    "frac": element_factory("mfrac", _arity=2),

# the base is the first argument
# the second argument is where effect applies
    "stackrel": element_factory("mover", _arity=2, _swap=True),
    "overset": element_factory("mover", _arity=2, _swap=True),
    "underset": element_factory("munder", _arity=2, _swap=True),

    "text": element_factory("mtext", _arity=1),
    "mbox": element_factory("mtext", _arity=1),

# sets mathcolor attrib

    "color": element_factory("mstyle", _arity=2, _o1_attr="mathcolor"),
    "cancel": element_factory("menclose", _arity=1, notation="updiagonalstrike"),

# new style tags
## bold
    "bb": element_factory("mstyle", _arity=1, fontweight="bold"),
    "mathbf": element_factory("mstyle", _arity=1, fontweight="bold"),

## sans
    "sf": element_factory("mstyle", _arity=1, fontfamily="sans"),
    "mathsf": element_factory("mstyle", _arity=1, fontfamily="sans"),

## double-struck
    "bbb": element_factory("mstyle", _arity=1, mathvariant="double-struck"),
    "mathbb": element_factory("mstyle", _arity=1, mathvariant="double-struck"),

## script
    "cc": element_factory("mstyle", _arity=1, mathvariant="script"),
    "mathcal": element_factory("mstyle", _arity=1, mathvariant="script"),

## monospace
    "tt": element_factory("mstyle", _arity=1, fontfamily="monospace"),
    "mathtt": element_factory("mstyle", _arity=1, fontfamily="monospace"),

## fraktur
    "fr": element_factory("mstyle", _arity=1, mathvariant="fraktur"),
    "mathfrak": element_factory("mstyle", _arity=1, mathvariant="fraktur"),
# {input:"mbox", tag:"mtext", output:"mbox", tex:null, ttype:TEXT},
# {input:"\"",   tag:"mtext", output:"mbox", tex:null, ttype:TEXT};
}

symbol_names = sorted(symbols.keys(), key=lambda s: len(s), reverse=True)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    from argparse import ArgumentParser

    aparser = ArgumentParser(
        usage='Test asciimathml with different etree elements'
    )
    text_modes = aparser.add_mutually_exclusive_group()
    text_modes.add_argument(
        '-m', '--markdown',
        default=False, action='store_true',
        help="Use markdown's etree element"
    )
    text_modes.add_argument(
        '-c', '--celement',
        default=False, action='store_true',
        help="Use cElementTree's element"
    )

    aparser.add_argument(
        'text',
        nargs='+',
        help='asciimath text to turn into mathml'
    )
    args_ns = aparser.parse_args(args)

    if args_ns.markdown:
        import markdown
        try:
            element = markdown.etree.Element
        except AttributeError as e:
            element = markdown.util.etree.Element
    elif args_ns.celement:
        from xml.etree.cElementTree import Element
        element = Element
    else:
        element = Element

    print("""\
<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
    <head>
        <meta http-equiv="Content-Type" content="application/xhtml+xml" />
        <title>ASCIIMathML preview</title>
    </head>
    <body>
""")
    result = parse(' '.join(args_ns.text), element)
    if sys.version_info.major >= 3:
        encoding = 'unicode'
    else:
        encoding = 'utf-8'
    print(tostring(result, encoding=encoding))
    print("""\
    </body>
</html>
""")


if __name__ == '__main__':
    main()
