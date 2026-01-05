# Dev log

## 2026-01-04

### Parser combinators for comptime

I am working on parser combinators in zig for comptime expression evaluation.
It's still a pleasure working with Zig :)

The way I've found to develop a parser lbrary this way is to start with very
simple forms. I'll write the code for some of the basic functions we want and
then as I need something new, I'll go back and refactor. Depending on the
language, I might end up doing things in different ways. For `zig`, it looks
like I can do parsers that are evaluable at `comptime`, which is crazy.

I like starting with:

```zig
fn take_while(input: []const u8, fn pred(input:[]const u8) bool) ?[]const u8
{
  ...
}
```

The idea is that this consumes any matching bytes from the input and returns
the remaining non-consumed byte if anything matches, and otherwise `null`.

That works fine for writing some simple functions that matchse whitespace,
digits, or a fixed target string. And you can combine them like so:

```zig
tag(digits(whitespace("  123helloworld").?).?,"hello").?; // returns "world"
```

But you don't get to see _what_ got matched at each step. To do that, we want
the parser functions to return something more than just the remaining slice:

```zig
struct ParseResult {
  match: []const u8,
  rest: []const u8
};
```

Next, we'll want a way to combine parsers so we can handle the matches. So how
do we do that? Notice, all the parsing functions take a similar form:

```zig
fn (input:[]const u8, ...parameters...) ParseResult;
```

If we can curry the parameters, then we are left with a uniform
`fn([]const u8) ParseResult`. It's pretty easy to do this especially because
the parameters for these functions are known at comptime. Creating the curried
closure looks something like:

```zig
fn Parser(comptime args: anytype, comptime func: anytype) type {
  return struct {
    fn parse(input: []const u8) anytype {
      return func(input,args);
    }
  }
}
```

Then we implement parsers like `take_while` so that they return the curried
function:

```zig
fn take_while(predicate fn(u8) bool) type {
  return Parser(predicate, struct {
    fn parse(input: []const u8, pred: @TypeOf(predicate)) ?ParseResult {
      ...
    }
  }.parse);
}
```

Now the parsers define functions that do the parsing, and we can combine those
to make new parsers. For example, this just works:

```zig
const whitespace = take_while(std.ascii.isWhitespace);
const digits = take_while(std.ascii.isDigit);
```

That ends up being much more succinct than writing these without the
functional approach.

The next step is to write some actual combinators: parsers constructed from
other parsers. For that, we'll want to generalize `ParseResult` so the value
returned by a match can vary depending on the parser. For example, when
matching a sequence of parsers, we'll want to return a tuple of matches as
the result. Changing things to return a `ParseResult(T)` is relatively
easy.

To write `seq`, we start with

```zig
fn seq(comptime parsers: anytype) type {...}
```

It takes `anytype` so that we can specify the parsers as a tuple. It's a
tuple because each `Parser` is actually a different concrete type. At comptime
we can inspect those types to compute what type of tuple should be returned
from the `seq` parser, `TupleType`:

```zig
fn seq(comptime parsers: anytype) type {
    ...
    // type introspection and calculation of TupleType
    ...
    return Parser(parser_types, struct {
        fn parse(input: Range, ps: @TypeOf(parser_types)) ?ParseResult(TupleType) {
            var current = input;
            var results: TupleType = undefined;

            inline for (ps, 0..) |ParserType, i| {
                const result = ParserType.parse(current) orelse return null;
                results[i] = result.match;
                current = result.rest;
            }

            return .{
                .match = results,
                .rest = current,
            };
        }
    }.parse);
}
```

A "type calculation" step tends to be needed for most of the combinators.
It's a kind of type inference you have to do yourself, and it definitely feels
pretty involved. However, you also get to do type checking here. You have
to do it yourself, but you have all the control. We can effectively have a
generic type system, but we only need as much as the application demands.

The downside is there's a lot of `type` and `anytype` running around. I can't
just look at the type signatures for these functions and figure out how they're
supposed to work. That's a little less true for libraries like `nom`. `zig`'s
typesystem doesn't have a declarative language around describing generic
types that we can rely on for this.

But I can get to this at the end of the day:

```zig
const number = recognize(seq(.{
    opt(tag("-")), // optional sign
    alt(Range, .{ tag("0"), digits }), // integer part ('0' or 1+ digits)
    opt(seq(.{ tag("."), digits })), // decimal part
    opt(seq(.{ tag("e"), opt(tag("-")), digits })), // optional exponent
}));
```

And I can evaluate it at comptime.

```zig
test "comptime number validation" {
    // Checks happen entirely at compile time!
    comptime {
        if (!isValidNumber("123.45")) {
            @compileError("123.45 should be a valid number");
        }
    }
}
```

### Back to arithmetic expressions

Can write the tokenizer like so:

```zig
const identifier = recognize(seq(.{ take_one(isAlpha), alphanumeric0 }));

const tokens = many(delimited(whitespace0, alt(.{
    value(TokenType.NUMBER, number),
    value(TokenType.IDENTIFIER, identifier),
    value(TokenType.PLUS, tag("+")),
    value(TokenType.MINUS, tag("-")),
    value(TokenType.STAR, tag("*")),
    value(TokenType.LPAREN, tag("(")),
    value(TokenType.RPAREN, tag(")")),
}), whitespace0));
```

The `many()` combinator is a bit tricky. It relies on comptime-only
accumulation of results. I need to look into alternatives since it would be
nice if I could use the same function for both runtime and comptime. At
runtime, it's probably possible to pass the allocator in with the parser
context.

Really the tokens should get mapped to a richer token type that retains the
matched slice.

This will produce an array of tokens. It would be nice to be able to reuse
the combinators to build parsers that work on top of the token stream to
ultimately build the AST.

Building the AST is another kind of accumulating construction, so I may
want a "fold" kind of method.


## 2026-01-02

### Multivector addition and product

Added a dense multivector type using `@Vector` for a simd-friendly representation.

Adds are trivial to vectorize in this way.

Vectorizing the multiplication is interesting. We have a relatively small and
well-ordered set of basis blades, $b_i$ we  need to worry about in any given
product. So we can compute a Cayley matrix $C_{ij}=b_i b_j$ at compile time. But
to compute the product of two multivectors $x$ and $y$, we need to add all the
$b_i$ and $b_j$ that end up contributing to the $k$-th blade of $xy$.

For a vectorized (SIMD) approach, that presents a challenge. We can vectorize
multiplying different pairs of blades together but they scatter to different
outputs. In fact, it turns out that the blade product $b_i b_j = s b_{i\oplus j}
= s b_k$ for some $s \in -1,0,1$ where $i\oplusj$ is an `xor`.

With the xor, we can invert the indexing: instead of scattering from $(i,j)$ to
$k = i \oplus j$, we compute each output $k$ by gathering from $i = k \oplus j$
across all $j$. This transforms the problem into a gather.

For each output coefficient: $z_k = \sum_j x_{k \oplus j} \cdot s_{k,j} \cdot
y_j$, where $s_{k,j}$ are precomputed signs. For $d$-dimensional multivectors
small enough to fit into a single simd register, this is $O(d 2^d)$.

This still isn't more efficient than the comparable products using matrix
representations. A matrix only has $d^2$ coefficients, where as a multivector
will have $2^d$! In that sense multivectors are a lot richer, but a bit
unwieldy in their dense form.

Certain objects (points, lines, etc) have sparse structure. Maybe products
can be made more efficient in those cases.

Might come back to this when implementing `join` or `meet`.

### Embedding a DSL in zig

I want to have a more ergonomic way of expressing computations with
multivectors or something, but Zig doesn't have operator overloading.

In rust you could use macros to make a kind of domain-specific language -
something that let's you programmatically generate code. Zig explicitly doesn't
do that. But I think you can use `comptime`.  An evaluator can generate an
AST for an expression and maybe even generate some pretty specific runtime
code.

I'll start by just doing full comptime evaluation. But it should give me
something like

```zig
eval("e0*(e21-e4+a)",.{a=4});
```

It would be cool if I could use that to define vectors like $e_+$ and $e_-$
from conformal geometric algebra.

I think I could also easily extend it to other products, add some simple
math functions (scalar sqrt, generalized exp), and enable runtime evaluation
of variables.

I made a start on this but I think I'll have to pick it up in the morning.

## 2026-01-01

Zig `comptime` is pretty good.

I've been able to use it to generalize the blade product. You can't really
capture the type requirements in the type signature. A generic type has to be
anytype. But you can check the invariants and generate a compiler error.

It's very much a duck-typing approach, so you have to check the invariants early
if you don't want to run into compiler errors more deeply into the code.

And you can dynamically generate structs. I used that to autogenerate the
basis blades for a particular space.

I don't think you can manipulate the ast, but you can fill out the metadata
for the type.
