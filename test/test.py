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

# -*- coding: utf-8 -*-

import unittest

from xml.etree.ElementTree import tostring

from asciimathml import parse, element_factory

import xml.dom.minidom as md

def pretty_print(s):
    return md.parseString(s).toprettyxml(indent='  ')

class ParseTestCase(unittest.TestCase):
    maxDiff = None

    def assertTreeEquals(self, a, b):
        ppa = pretty_print(tostring(a))
        ppb = pretty_print(tostring(b))
        # open('got.xml', 'w').write(ppa.encode('utf-8'))
        # open('expected.xml', 'w').write(ppb.encode('utf-8'))
        self.assertEquals(ppa, ppb)

    def assertRendersTo(self, asciimathml, xmlstring):
        mathml = parse(asciimathml)
        ppa = pretty_print(tostring(mathml))
        ppb = pretty_print('<math><mstyle>%s</mstyle></math>' % xmlstring)
        # open('got.xml', 'w').write(ppa.encode('utf-8'))
        # open('expected.xml', 'w').write(ppb.encode('utf-8'))
        self.assertEquals(ppa, ppb)

    def testEmpty(self):
        self.assertTreeEquals(parse(''), element_factory('math', element_factory('mstyle')))

    def testNumber(self):
        self.assertTreeEquals(
            parse('3.1415'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mn', text='3.1415')
                )
            )
        )

    def testSymbol(self):
        self.assertTreeEquals(
            parse('alpha'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mi', text=u'\u03b1')
                )
            )
        )

    def testSymbols(self):
        self.assertTreeEquals(
            parse('alpha beta'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mi', text=u'\u03b1'),
                    element_factory('mi', text=u'\u03b2')
                )
            )
        )

    def testFrac(self):
        self.assertTreeEquals(
            parse('alpha / beta'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mfrac',
                        element_factory('mi', text=u'\u03b1'),
                        element_factory('mi', text=u'\u03b2')
                    )
                )
            )
        )

    def testText(self):
        self.assertTreeEquals(
            parse('text{undefined}'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mrow',
                        element_factory('mtext', text='undefined')
                    )
                )
            )
        )

    def testQuotedText(self):
        self.assertTreeEquals(
            parse('"time" = "money"'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mrow',
                        element_factory('mtext', text='time')
                    ),
                    element_factory('mo', text='='),
                    element_factory('mrow',
                        element_factory('mtext', text='money')
                    )
                )
            )
        )

    def testQuotedSpaces(self):
        self.assertTreeEquals(
            parse('" a  b c   " x'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mrow',
                        element_factory('mspace', width='1ex'),
                        element_factory('mtext', text=' a  b c   '),
                        element_factory('mspace', width='1ex'),
                    ),
                    element_factory('mi', text='x'),
                )
            )
        )

    def testIncompleteFrac(self):
        self.assertTreeEquals(
            parse('alpha /'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mfrac',
                        element_factory('mi', text=u'\u03b1'),
                        element_factory('mo')
                    )
                )
            )
        )

    def testDivision(self):
        self.assertTreeEquals(
            parse('alpha // beta'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mi', text=u'\u03b1'),
                    element_factory('mo', text='/'),
                    element_factory('mi', text=u'\u03b2')
                )
            )
        )

    def testSub(self):
        self.assertTreeEquals(
            parse('alpha _ beta'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('msub',
                        element_factory('mi', text=u'\u03b1'),
                        element_factory('mi', text=u'\u03b2')
                    )
                )
            )
        )

    def testSup(self):
        self.assertTreeEquals(
            parse('alpha ^ beta'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('msup',
                        element_factory('mi', text=u'\u03b1'),
                        element_factory('mi', text=u'\u03b2')
                    )
                )
            )
        )

    def testSubSup(self):
        self.assertTreeEquals(
            parse('alpha _ beta ^ gamma'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('msubsup',
                        element_factory('mi', text=u'\u03b1'),
                        element_factory('mi', text=u'\u03b2'),
                        element_factory('mi', text=u'\u03b3')
                    )
                )
            )
        )

    def testSupSub(self):
        self.assertTreeEquals(parse('alpha ^ beta _ gamma'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('msubsup',
                        element_factory('mi', text=u'\u03b1'),
                        element_factory('mi', text=u'\u03b3'),
                        element_factory('mi', text=u'\u03b2')
                    )
                )
            )
        )

    def testUnary(self):
        self.assertTreeEquals(
            parse('sin alpha'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mrow',
                        element_factory('mo', text='sin'),
                        element_factory('mi', text=u'\u03b1')
                    )
                )
            )
        )

    def testUnary2(self):
        self.assertTreeEquals(
            parse('dot alpha'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mover',
                        element_factory('mi', text=u'\u03b1'),
                        element_factory('mo', text='.'))
                )
            )
        )

    def testUnary3(self):
        self.assertTreeEquals(
            parse('sqrt alpha'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('msqrt',
                        element_factory('mi', text=u'\u03b1')
                    )
                )
            )
        )

    def testUnary4(self):
        self.assertTreeEquals(
            parse('text alpha'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mtext',
                        element_factory('mi', text=u'\u03b1')
                    )
                )
            )
        )

    def testBinary(self):
        self.assertTreeEquals(
            parse('frac alpha beta'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mfrac',
                        element_factory('mi', text=u'\u03b1'),
                        element_factory('mi', text=u'\u03b2')
                    )
                )
            )
        )

    # asciimathml.js somehow interpret a/b/c/d as (a/b)/(c/d)
    # Our python implementation requires to be explicit in this case, but
    # I'm not sure this is a problem
    @unittest.expectedFailure
    def testTripleFrac(self):
        self.assertRendersTo(
            'a/b/c/d',
            '<mfrac>'
                '<mi>a</mi>'
                '<mi>b</mi>'
            '</mfrac>'
            '<mo>/</mo>'
            '<mfrac>'
                '<mi>c</mi>'
                '<mi>d</mi>'
            '</mfrac>'
        )

    def testUnderOver(self):
        self.assertTreeEquals(
            parse('sum_alpha^beta'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('munderover',
                        element_factory('mo', text=u'\u2211'),
                        element_factory('mi', text=u'\u03b1'),
                        element_factory('mi', text=u'\u03b2')
                    )
                )
            )
        )

    def testColor(self):
        self.assertTreeEquals(
            parse('color (blue) x'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mstyle',
                        element_factory('mi', text=u'x'),
                        mathcolor='blue'
                    )
                )
            )
        )

    def testParens(self):
        self.assertTreeEquals(
            parse('(alpha + beta) / gamma'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mfrac',
                        element_factory('mrow',
                            element_factory('mi', text=u'\u03b1'),
                            element_factory('mo', text='+'),
                            element_factory('mi', text=u'\u03b2')),
                        element_factory('mi', text=u'\u03b3')
                    )
                )
            )
        )

    def testUnbalancedParens(self):
        self.assertRendersTo(
            '(alpha + beta / gamma',
            '<mrow>'
                '<mo>(</mo>'
                '<mi>&#945;</mi>'
                '<mo>+</mo>'
                '<mfrac>'
                    '<mi>&#946;</mi>'
                    '<mi>&#947;</mi>'
                '</mfrac>'
            '</mrow>'
        )

    def testNestedParens(self):
        self.assertRendersTo(
            '(alpha + [ beta + { gamma } ] )',
            '<mrow>'
                '<mo>(</mo>'
                '<mi>&#945;</mi>'
                '<mo>+</mo>'
                '<mrow>'
                    '<mo>[</mo>'
                    '<mi>&#946;</mi>'
                    '<mo>+</mo>'
                    '<mrow>'
                        '<mo>{</mo>'
                        '<mi>&#947;</mi>'
                        '<mo>}</mo>'
                    '</mrow>'
                    '<mo>]</mo>'
                '</mrow>'
                '<mo>)</mo>'
            '</mrow>'
        )

    def testNesting(self):
        self.assertRendersTo(
            'sqrt sqrt root3x',
            '<msqrt>'
                '<msqrt>'
                    '<mroot>'
                        '<mi>x</mi>'
                        '<mn>3</mn>'
                    '</mroot>'
                '</msqrt>'
            '</msqrt>'
        )

    # Here the bar '|' should be treated as an operator (maybe wrapped in an
    # mrow, but I'm not sure that's needed).  Instead we are treating it as an
    # unclosed delimiter which we close implicitly at the next }
    def testBar(self):
        self.assertRendersTo(
            '{x | y}',
            '<mrow>'
                '<mo>{</mo>'
                '<mi>x</mi>'
                '<mo>|</mo>'
                '<mi>y</mi>'
                '<mo>}</mo>'
            '</mrow>'
        )

    def testNegative(self):
        self.assertRendersTo(
            'abc-123.45^-1.1',
            '<mi>a</mi>'
            '<mi>b</mi>'
            '<mi>c</mi>'
            '<mo>-</mo>'
            '<msup>'
                '<mn>123.45</mn>'
                '<mrow>'
                    '<mo>-</mo>'
                    '<mn>1.1</mn>'
                '</mrow>'
            '</msup>'
        )

    def testHat(self):
        self.assertRendersTo(
            'hat(ab) bar(xy) ulA vec v dotx ddot y',
            '<mover>'
                '<mrow>'
                    '<mi>a</mi>'
                    '<mi>b</mi>'
                '</mrow>'
                '<mo>^</mo>'
            '</mover>'
            '<mover>'
                '<mrow>'
                    '<mi>x</mi>'
                    '<mi>y</mi>'
                '</mrow>'
                '<mo>&#175;</mo>'
            '</mover>'
            '<munder>'
                '<mi>A</mi>'
                '<mo>&#818;</mo>'
            '</munder>'
            '<mover>'
                '<mi>v</mi>'
                '<mo>&#8594;</mo>'
            '</mover>'
            '<mover>'
                '<mi>x</mi>'
                '<mo>.</mo>'
            '</mover>'
            '<mover>'
                '<mi>y</mi>'
                '<mo>..</mo>'
            '</mover>'
        )

    def testMatrix(self):
        self.assertRendersTo(
            '[[a,b],[c,d]]((n),(k))',
            '<mrow>'
                '<mo>[</mo>'
                '<mtable>'
                    '<mtr>'
                        '<mtd>'
                            '<mi>a</mi>'
                        '</mtd>'
                        '<mtd>'
                            '<mi>b</mi>'
                        '</mtd>'
                    '</mtr>'
                    '<mtr>'
                        '<mtd>'
                            '<mi>c</mi>'
                        '</mtd>'
                        '<mtd>'
                            '<mi>d</mi>'
                        '</mtd>'
                    '</mtr>'
                '</mtable>'
                '<mo>]</mo>'
            '</mrow>'
            '<mrow>'
                '<mo>(</mo>'
                '<mtable>'
                    '<mtr>'
                        '<mtd>'
                            '<mi>n</mi>'
                        '</mtd>'
                    '</mtr>'
                    '<mtr>'
                        '<mtd>'
                            '<mi>k</mi>'
                        '</mtd>'
                    '</mtr>'
                '</mtable>'
                '<mo>)</mo>'
            '</mrow>'
        )

    def testMatrix2(self):
        self.assertRendersTo(
            'x/x={(1,if x!=0),(text{undefined},if x=0):}', # columnalign="left"
            '<mfrac><mi>x</mi><mi>x</mi></mfrac>'
            '<mo>=</mo>'
            '<mrow>'
                '<mo>{</mo>'
                '<mtable>'
                '<mtr>'
                    '<mtd><mn>1</mn></mtd>'
                    '<mtd>'
                        '<mrow><mspace width="1ex" /><mo>if</mo><mspace width="1ex" /></mrow>'
                        '<mi>x</mi>'
                        '<mo>&#8800;</mo>'
                        '<mn>0</mn>'
                    '</mtd>'
                '</mtr>'
                '<mtr>'
                    '<mtd>'
                        '<mrow><mtext>undefined</mtext></mrow>'
                    '</mtd>'
                    '<mtd>'
                        '<mrow>'
                            '<mspace width="1ex" />'
                            '<mo>if</mo>'
                            '<mspace width="1ex" />'
                        '</mrow>'
                        '<mi>x</mi>'
                        '<mo>=</mo>'
                        '<mn>0</mn>'
                    '</mtd>'
                '</mtr>'
                '</mtable>'
            '</mrow>'
        )

    # asciimathml.js wraps twice the `x` with a single bar (<mrow><mo>|</mo><mrow><mo>|</mo> ...)
    # we use a double bar instead
    def testDoubleBar(self):
        self.assertRendersTo(
            '||x||^2',
            '<msup>'
                '<mrow>'
                    '<mo>&#x2016;</mo>'
                    '<mi>x</mi>'
                    '<mo>&#x2016;</mo>'
                '</mrow>'
                '<mn>2</mn>'
            '</msup>'
        )

    def testRewriteLRAdditive(self):
        self.assertTreeEquals(
            parse('floor A'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mrow',
                        element_factory('mo', u"\u230A"),
                        element_factory('mi', 'A'),
                        element_factory('mo', u"\u230B")
                    )
                )
            )
        )

    def tesRewriteLRNested(self):
        self.assertTreeEquals(
            parse('floor abs A'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mrow',
                        element_factory('mo', u"\u230A"),
                        element_factory('mrow',
                            element_factory('mo', '|'),
                            element_factory('mi', 'A'),
                            element_factory('mo', '|')
                        ),
                        element_factory('mo', u"\u230B")
                    )
                )
            )
        )

    def testRewriteLRReplace(self):
        self.assertTreeEquals(
            parse('abs(xyz)'),
            element_factory('math',
                element_factory('mstyle',
                    element_factory('mrow',
                        element_factory('mo', u"|"),
                        element_factory('mrow',
                            element_factory('mi', 'x'),
                            element_factory('mi', 'y'),
                            element_factory('mi', 'z'),
                        ),
                        element_factory('mo', u"|")
                    )
                )
            )
        )

if __name__ == '__main__':
    unittest.main()

