# Release Notes

### 0.9.5

#### Features
- Added Travis CI build
- Added Python 3 support
- Added new symbol mappings from asciimathml's javascript implementation
- Added `tilde` and `~` operators for putting ~'s above items'
- Added rewrite parentheses operators `abs`, `floor`, `norm`, and `ceil`
- Added font commands `bb`, `sf`, `bbb`, `cc`, `tt`, and `fr`
- Added `color` operator for setting color
- Added explicit `overset` and `underset` stacking operators
- Support quoted strings

#### Fixes
- Reduce the length of lines wherever possible
- Fix remove_invisible not working, making `{:` and `:}` unusable
- Do not surround text operator in an mrow for certain cases,
  such as forcing `alpha` to render as text.
- Incomplete special binaries, such as `/`,
  now render their partially filled XML
- Do not improperly terminate an mrow when
  a lone symmetric parenthesis (`|` or `||`) is
  nested inside another set of parentheses.
- Actually treat lone symmetric parentheses as an infix operator.
- Unpack `-` signs if there's a term before the negative term.

### 0.9.4

Added double bar delimiters (for example ||x||).  Note that the bars are
rendered with a unicode double bar, whereas asciimathml.js uses a double single
bar.

### 0.9.3

Added tests, license, README and RELEASE-NOTES to the distribution.
