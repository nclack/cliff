const std = @import("std");
const cliff = @import("cliff");

fn ParseResult(T: type) type {
    return struct {
        match: T,
        rest: []const u8,
    };
}

/// Creates a Parser type from a compile-time closure.
///
/// Contract:
/// - `args`: Any compile-time value to capture (e.g., a predicate function, a string, etc.)
/// - `func`: A function with signature `fn([]const u8, @TypeOf(args)) ?ParseResult(T)` for some type T
///
/// The returned Parser type has a `parse([]const u8)` method that returns `?ParseResult(T)`.
///
/// Example:
///   ```zig
///   const my_parser = Parser("hello", struct {
///       fn parse(input: []const u8, target: []const u8) ?ParseResult([]const u8) {
///           // ... parsing logic ...
///       }
///   }.parse);
///   ```
fn Parser(comptime args: anytype, comptime func: anytype) type {
    // Extract the return type of the function
    const func_type_info = @typeInfo(@TypeOf(func));

    // Handle both function pointers and direct functions
    const ReturnType = switch (func_type_info) {
        .@"fn" => |fn_info| fn_info.return_type.?,
        .pointer => |ptr_info| blk: {
            const pointee_info = @typeInfo(ptr_info.child);
            if (pointee_info == .@"fn") {
                break :blk pointee_info.@"fn".return_type.?;
            }
            @compileError("Expected function or function pointer");
        },
        else => @compileError("Expected function or function pointer"),
    };

    // compile-time type validation
    comptime {
        // Check it's an optional
        const type_info = @typeInfo(ReturnType);
        if (type_info != .optional) {
            @compileError("Parser function must return an optional type");
        }

        // Check the child has match and rest fields
        const ChildType = type_info.optional.child;
        const child_info = @typeInfo(ChildType);
        if (child_info != .@"struct") {
            @compileError("Parser result must be a ParseResult");
        }

        // Verify required fields exist
        var has_match = false;
        var has_rest = false;
        for (child_info.@"struct".fields) |field| {
            if (std.mem.eql(u8, field.name, "match")) has_match = true;
            if (std.mem.eql(u8, field.name, "rest")) has_rest = true;
        }
        if (!has_match or !has_rest) {
            @compileError("Parser result must be a ParseResult");
        }
    }

    return struct {
        const Self = @This();

        fn parse(input: []const u8) ReturnType {
            return func(input, args);
        }
    };
}

/// Returns a Parser that matches while the predicate is true.
/// NOTE: Relies on Utf8Iterator.i pointing to the byte position
///       after the last consumed codepoint.
/// NOTE: zig decodes utf-8 codepoints to u21.
fn take_while(predicate: fn (u21) bool) type {
    return Parser(predicate, struct {
        fn parse(input: []const u8, predicate_: fn (u21) bool) ?ParseResult([]const u8) {
            var last_match_pos: ?usize = null;
            var it = std.unicode.Utf8Iterator{ .bytes = input, .i = 0 };
            while (it.nextCodepoint()) |codepoint| {
                if (predicate_(codepoint)) {
                    // it.i points past the consumed bytes
                    last_match_pos = it.i;
                } else {
                    break;
                }
            }
            return if (last_match_pos) |i|
                .{ .match = input[0..i], .rest = input[i..] }
            else
                null;
        }
    }.parse);
}

fn take_one(predicate: fn (u21) bool) type {
    return Parser(predicate, struct {
        fn parse(input: []const u8, predicate_: fn (u21) bool) ?ParseResult([]const u8) {
            var last_match_pos: ?usize = null;
            var it = std.unicode.Utf8Iterator{ .bytes = input, .i = 0 };
            if (it.nextCodepoint()) |codepoint| {
                if (predicate_(codepoint)) {
                    // it.i points past the consumed bytes
                    last_match_pos = it.i;
                }
            }
            return if (last_match_pos) |i|
                .{ .match = input[0..i], .rest = input[i..] }
            else
                null;
        }
    }.parse);
}

fn isWhitespace(codepoint: u21) bool {
    // This works because any non-ascii character will have the most
    // significant bit set in the low byte.
    return std.ascii.isWhitespace(@truncate(codepoint));
}

fn isDigit(codepoint: u21) bool {
    // This works because any non-ascii character will have the most
    // significant bit set in the low byte.
    return std.ascii.isDigit(@truncate(codepoint));
}

fn isAlpha(codepoint: u21) bool {
    // This works because any non-ascii character will have the most
    // significant bit set in the low byte.
    return std.ascii.isAlphabetic(@truncate(codepoint));
}

fn isAlphanumeric(codepoint: u21) bool {
    // This works because any non-ascii character will have the most
    // significant bit set in the low byte.
    return std.ascii.isAlphanumeric(@truncate(codepoint));
}

const whitespace = take_while(isWhitespace);
const whitespace0 = opt(whitespace);
const alphanumeric = take_while(isAlphanumeric);
const alphanumeric0 = opt(alphanumeric);
const digits = take_while(isDigit);

/// Returns a Parser that matches the given target string.
fn tag(target: []const u8) type {
    return Parser(target, struct {
        fn parse(input: []const u8, target_: []const u8) ?ParseResult([]const u8) {
            // This works for utf-8 since we're just comparing byte slices.
            if (input.len >= target_.len) {
                if (std.mem.eql(u8, input[0..target_.len], target_)) {
                    return .{ .match = input[0..target_.len], .rest = input[target_.len..] };
                }
            }
            return null;
        }
    }.parse);
}

/// Returns a Parser that optionally applies the given parser.
/// Always succeeds - returns the parse result if successful, or null match if not.
fn opt(comptime parser: type) type {
    // Extract the match type from the parser
    const parse_return_type = @typeInfo(@TypeOf(parser.parse)).@"fn".return_type.?;
    const parse_result_type = @typeInfo(parse_return_type).optional.child;
    const fields = @typeInfo(parse_result_type).@"struct".fields;
    comptime var match_type: type = undefined;
    for (fields) |field| {
        if (std.mem.eql(u8, field.name, "match")) {
            match_type = field.type;
            break;
        }
    }

    return Parser(parser, struct {
        fn parse(input: []const u8, parser_: type) ?ParseResult(?match_type) {
            if (parser_.parse(input)) |result| {
                return .{
                    .match = result.match,
                    .rest = result.rest,
                };
            }
            // successful, but empty, match
            return .{ .match = null, .rest = input };
        }
    }.parse);
}

/// Returns a Parser that tests a list of parsers until one succeeds.
/// All parsers must return ?ParseResult(T) for the same type T.
///
/// Example: `alt([]const u8, .{ tag("hello"), tag("goodbye") })`
fn alt(comptime T: type, comptime parsers: anytype) type {

    // TODO: Just extract the match type from the first parser, and encorce that all parsers have the same match type

    const parsers_type_info = @typeInfo(@TypeOf(parsers));

    // Convert tuple to array of types
    const parser_types = switch (parsers_type_info) {
        .@"struct" => |struct_info| blk: {
            if (!struct_info.is_tuple) {
                @compileError("Expected tuple of parser types, e.g., .{ tag(\"hello\"), tag(\"goodbye\") }");
            }
            comptime var types: [struct_info.fields.len]type = undefined;
            inline for (struct_info.fields, 0..) |field, i| {
                types[i] = @field(parsers, field.name);
            }
            break :blk types;
        },
        .pointer => |ptr_info| blk: {
            if (ptr_info.size == .Slice and ptr_info.child == type) {
                // Also accept slice syntax for backwards compatibility
                break :blk parsers;
            }
            @compileError("Expected tuple of parser types");
        },
        else => @compileError("Expected tuple of parser types, e.g., .{ tag(\"hello\"), tag(\"goodbye\") }"),
    };

    return Parser(parser_types, struct {
        fn parse(input: []const u8, ps: @TypeOf(parser_types)) ?ParseResult(T) {
            inline for (ps) |ParserType| {
                if (ParserType.parse(input)) |result| {
                    return result;
                }
            }
            return null;
        }
    }.parse);
}

/// Returns a Parser that matches a sequence of parsers and returns a tuple of results.
/// Each parser is applied in order, and all must succeed for seq to succeed.
/// Returns ?ParseResult(Tuple) where Tuple contains all the match values.
///
/// Example: `seq(.{ tag("hello"), whitespace, tag("world") })`
fn seq(comptime parsers: anytype) type {
    const parsers_type_info = @typeInfo(@TypeOf(parsers));

    // Convert tuple to array of types
    const parser_types = switch (parsers_type_info) {
        .@"struct" => |struct_info| blk: {
            if (!struct_info.is_tuple) {
                @compileError("Expected tuple of Parser types, e.g., .{ tag(\"hello\"), whitespace }");
            }
            comptime var types: [struct_info.fields.len]type = undefined;
            inline for (struct_info.fields, 0..) |field, i| {
                types[i] = @field(parsers, field.name);
            }
            break :blk types;
        },
        .pointer => |ptr_info| blk: {
            if (ptr_info.size == .Slice and ptr_info.child == type) {
                // Also accept slice syntax for backwards compatibility
                break :blk parsers;
            }
            @compileError("Expected tuple of Parser types");
        },
        else => @compileError("Expected tuple of Parser types, e.g., .{ tag(\"hello\"), whitespace }"),
    };

    // Extract the match type from each parser
    comptime var match_types: [parser_types.len]type = undefined;
    inline for (parser_types, 0..) |ParserType, i| {
        // Get parse function return type: ?ParseResult(T)
        const parse_return_type = @typeInfo(@TypeOf(ParserType.parse)).@"fn".return_type.?;
        // Unwrap optional: ParseResult(T)
        const parse_result_type = @typeInfo(parse_return_type).optional.child;
        // Get the match field type: T
        const fields = @typeInfo(parse_result_type).@"struct".fields;
        for (fields) |field| {
            if (std.mem.eql(u8, field.name, "match")) {
                match_types[i] = field.type;
                break;
            }
        }
    }

    // Create a tuple type from all the match types
    const TupleType = std.meta.Tuple(&match_types);

    return Parser(parser_types, struct {
        fn parse(input: []const u8, ps: @TypeOf(parser_types)) ?ParseResult(TupleType) {
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

/// If the child parser was successful, return the consumed input as produced value.
fn recognize(comptime parser: type) type {
    return Parser(parser, struct {
        fn parse(input: []const u8, parser_: type) ?ParseResult([]const u8) {
            const result = parser_.parse(input) orelse return null;
            const consumed = input.len - result.rest.len;
            return .{
                .match = input[0..consumed],
                .rest = result.rest,
            };
        }
    }.parse);
}

const number = recognize(seq(.{
    opt(tag("-")), // optional sign
    alt([]const u8, .{ tag("0"), digits }), // integer part ('0' or 1+ digits)
    opt(seq(.{ tag("."), digits })), // decimal part
    opt(seq(.{ tag("e"), opt(tag("-")), digits })), // optional exponent
}));

/// Maps the match type for the parser
fn map(comptime mapfn: anytype, comptime parser: type) type {
    // 1. Extract the match type from the parser
    const parse_return_type = @typeInfo(@TypeOf(parser.parse)).@"fn".return_type.?;
    const parse_result_type = @typeInfo(parse_return_type).optional.child;
    const fields = @typeInfo(parse_result_type).@"struct".fields;
    comptime var match_type: type = undefined;
    for (fields) |field| {
        if (std.mem.eql(u8, field.name, "match")) {
            match_type = field.type;
            break;
        }
    }

    // 2. Validate and extract mapfn information
    const mapfn_info = @typeInfo(@TypeOf(mapfn));
    if (mapfn_info != .@"fn") {
        @compileError("map expects a function");
    }

    const params = mapfn_info.@"fn".params;
    if (params.len != 1) {
        @compileError("map function must have exactly 1 parameter");
    }

    const param_type = params[0].type.?;
    if (param_type != match_type) {
        @compileError("map function parameter type must match parser match type");
    }

    const MappedType = mapfn_info.@"fn".return_type.?;

    // 3. Create and return the mapped parser
    return Parser(.{ .parser = parser, .mapfn = mapfn }, struct {
        fn parse(input: []const u8, ctx: anytype) ?ParseResult(MappedType) {
            const result = ctx.parser.parse(input) orelse return null;
            return .{
                .match = ctx.mapfn(result.match),
                .rest = result.rest,
            };
        }
    }.parse);
}

/// Matches the input parser 1 or more times, returning an array of matches.
/// Fails if the parser doesn't match at least once.
/// Comptime-only (builds the array during compilation).
fn many(comptime parser: type) type {
    // Extract the match type from the parser
    const parse_return_type = @typeInfo(@TypeOf(parser.parse)).@"fn".return_type.?;
    const parse_result_type = @typeInfo(parse_return_type).optional.child;
    const fields = @typeInfo(parse_result_type).@"struct".fields;
    comptime var match_type: type = undefined;
    for (fields) |field| {
        if (std.mem.eql(u8, field.name, "match")) {
            match_type = field.type;
            break;
        }
    }

    return Parser(parser, struct {
        fn parse(input: []const u8, parser_: type) ?ParseResult([]const match_type) {
            var results: []const match_type = &.{};
            var current = input;
            var iterations: usize = 0;
            const max_iterations = 1000;

            while (iterations < max_iterations) : (iterations += 1) {
                if (parser_.parse(current)) |result| {
                    results = results ++ &[_]match_type{result.match};
                    current = result.rest;
                } else {
                    break;
                }
            }

            // Require at least one match
            if (results.len == 0) {
                return null;
            }

            return .{
                .match = results,
                .rest = current,
            };
        }
    }.parse);
}

/// Returns the provided value if the child parser succeeds.
fn value(comptime val: anytype, comptime parser: type) type {
    return Parser(.{ .parser = parser, .value = val }, struct {
        fn parse(input: []const u8, ctx: anytype) ?ParseResult(@TypeOf(val)) {
            const result = ctx.parser.parse(input) orelse return null;
            return .{
                .match = val,
                .rest = result.rest,
            };
        }
    }.parse);
}

// FIXME: delimited, terminated, preceded could probably be written as a map(seq(...))

/// Matches an object from the first parser and discards it, then gets an object
/// from the second parser, and finally matches an object from the third parser
/// and discards it.
fn delimited(comptime FirstParser: type, comptime SecondParser: type, comptime ThirdParser: type) type {

    // 1. Extract the match type from the SecondParser
    const parse_return_type = @typeInfo(@TypeOf(SecondParser.parse)).@"fn".return_type.?;
    const parse_result_type = @typeInfo(parse_return_type).optional.child;
    const fields = @typeInfo(parse_result_type).@"struct".fields;
    comptime var match_type: type = undefined;
    for (fields) |field| {
        if (std.mem.eql(u8, field.name, "match")) {
            match_type = field.type;
            break;
        }
    }

    // 2. Make the parser
    return Parser(.{ FirstParser, SecondParser, ThirdParser }, struct {
        fn parse(input: []const u8, ctx: anytype) ?ParseResult(match_type) {
            const result = seq(ctx).parse(input) orelse return null;
            return .{
                .match = result.match[1],
                .rest = result.rest,
            };
        }
    }.parse);
}

/// Gets an object from the Value parser, and matches an object from the
/// Terminator parser and discards it.
fn terminated(comptime ValueParser: type, comptime TerminatorParser: type) type {

    // 1. Extract the match type from the SecondParser
    const parse_return_type = @typeInfo(@TypeOf(ValueParser.parse)).@"fn".return_type.?;
    const parse_result_type = @typeInfo(parse_return_type).optional.child;
    const fields = @typeInfo(parse_result_type).@"struct".fields;
    comptime var match_type: type = undefined;
    for (fields) |field| {
        if (std.mem.eql(u8, field.name, "match")) {
            match_type = field.type;
            break;
        }
    }

    // 2. Make the parser
    return Parser(.{ ValueParser, TerminatorParser }, struct {
        fn parse(input: []const u8, ctx: anytype) ?ParseResult(match_type) {
            const result = seq(ctx).parse(input) orelse return null;
            return .{
                .match = result.match[0],
                .rest = result.rest,
            };
        }
    }.parse);
}

// === Expression language: arithmetic expressions - Tokenizer ===

/// Token types for arithmetic expressions
pub const TokenType = enum {
    NUMBER,
    IDENTIFIER,
    PLUS,
    MINUS,
    STAR,
    LPAREN,
    RPAREN,
    INVALID,
};

const Token = struct {
    type: TokenType,
    lexeme: []const u8,
};

fn mk_token(comptime kind: TokenType) fn ([]const u8) Token {
    return struct {
        fn inner(lexeme: []const u8) Token {
            return .{ .type = kind, .lexeme = lexeme };
        }
    }.inner;
}

const identifier = recognize(seq(.{ take_one(isAlpha), alphanumeric0 }));

const tokenize = many(delimited(whitespace0, alt(Token, .{
    map(mk_token(TokenType.NUMBER), number),
    map(mk_token(TokenType.NUMBER), number),
    map(mk_token(TokenType.IDENTIFIER), identifier),
    map(mk_token(TokenType.PLUS), tag("+")),
    map(mk_token(TokenType.MINUS), tag("-")),
    map(mk_token(TokenType.STAR), tag("*")),
    map(mk_token(TokenType.LPAREN), tag("(")),
    map(mk_token(TokenType.RPAREN), tag(")")),
    // Invalid tokens
    // Try to consume any non-alphanumeric characters first.
    // If that doesn't work then just take everything up to
    // the next whitespace
    map(mk_token(TokenType.INVALID), take_while(struct {
        fn inner(codepoint: u21) bool {
            return !(isAlphanumeric(codepoint) or isWhitespace(codepoint));
        }
    }.inner)),
    map(mk_token(TokenType.INVALID), take_while(struct {
        fn inner(codepoint: u21) bool {
            return !isWhitespace(codepoint);
        }
    }.inner)),
}), whitespace0));

// === TODO: convert below to use parser combinators. ===

/// A token with position information
pub const OpType = enum {
    ADD,
    SUB,
    MUL,
};

pub const ASTNode = union(enum) {
    Scalar: f32, // parsed scalar value
    BinOp: struct {
        op: OpType,
        lhs: *const ASTNode,
        rhs: *const ASTNode,
    },

    pub fn format(self: ASTNode, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        switch (self) {
            .Scalar => |s| try writer.print("Scalar({d})", .{s}),
            .BinOp => |b| {
                const op_str = switch (b.op) {
                    .ADD => "+",
                    .SUB => "-",
                    .MUL => "*",
                };
                try writer.print("({} {} {})", .{ b.lhs.*, op_str, b.rhs.* });
            },
        }
    }
};

pub const OldParseError = struct {
    message: []const u8,
    pos: usize,
};

pub const OldParseResult = struct {
    node: ?ASTNode,
    error_info: ?OldParseError,
    pos: usize, // current position in token stream
};

// ============================================================================
// Evaluator
// ============================================================================

/// Evaluate an AST node to a Multivector
fn evaluateNode(comptime BasisType: type, comptime node: ASTNode, vars: anytype) BasisType.Multivector {
    switch (node) {
        .Scalar => |val| {
            // Value is already parsed by the parser
            var result = BasisType.Multivector.zero();
            result.coefficients[0] = val;
            return result;
        },
        .BinOp => |binop| {
            const lhs = evaluateNode(BasisType, binop.lhs.*, vars);
            const rhs = evaluateNode(BasisType, binop.rhs.*, vars);

            return switch (binop.op) {
                .ADD => lhs.add(rhs),
                .MUL => lhs.mul(rhs),
            };
        },
    }
}

// Main eval function: tokenize, parse, and evaluate an expression at comptime

// pub fn eval(basis: anytype, comptime expr: []const u8, vars: anytype) @TypeOf(basis).Multivector {
//     comptime {
//         const tokens = tokenize(expr);
//         const parse_result = parse(tokens);

//         if (parse_result.error_info) |err| {
//             @compileError("Parse error at position " ++ std.fmt.comptimePrint("{d}", .{err.pos}) ++ ": " ++ err.message);
//         }

//         const ast = parse_result.node orelse @compileError("No AST generated");
//         return evaluateNode(@TypeOf(basis), ast, vars);
//     }
// }

// ============================================================================
// Tests
// ============================================================================

test "tokenize: single number" {
    const tokens = comptime tokenize.parse("42").?.match;
    try std.testing.expectEqual(1, tokens.len);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[0].type);
    try std.testing.expectEqualStrings("42", tokens[0].lexeme);
}

test "tokenize: decimal number" {
    const tokens = comptime tokenize.parse("3.14").?.match;
    try std.testing.expectEqual(1, tokens.len);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[0].type);
    try std.testing.expectEqualStrings("3.14", tokens[0].lexeme);
}

test "tokenize: addition" {
    const tokens = comptime tokenize.parse("1+2").?.match;
    try std.testing.expectEqual(3, tokens.len);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[0].type);
    try std.testing.expectEqualStrings("1", tokens[0].lexeme);
    try std.testing.expectEqual(TokenType.PLUS, tokens[1].type);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[2].type);
    try std.testing.expectEqualStrings("2", tokens[2].lexeme);
}

test "tokenize: multiplication with spaces" {
    const tokens = comptime tokenize.parse("2 * 3").?.match;
    try std.testing.expectEqual(3, tokens.len);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[0].type);
    try std.testing.expectEqual(TokenType.STAR, tokens[1].type);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[2].type);
}

test "tokenize: parentheses" {
    const tokens = comptime tokenize.parse("(1+2)").?.match;
    try std.testing.expectEqual(5, tokens.len);
    try std.testing.expectEqual(TokenType.LPAREN, tokens[0].type);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[1].type);
    try std.testing.expectEqual(TokenType.PLUS, tokens[2].type);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[3].type);
    try std.testing.expectEqual(TokenType.RPAREN, tokens[4].type);
}

test "tokenize: invalid character" {
    const tokens = comptime tokenize.parse("1@2").?.match;
    try std.testing.expectEqual(3, tokens.len);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[0].type);
    try std.testing.expectEqual(TokenType.INVALID, tokens[1].type);
    try std.testing.expectEqualStrings("@", tokens[1].lexeme);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[2].type);
}

// test "parse: single number" {
//     const tokens = comptime tokenize("42");
//     const result = comptime parse(tokens);
//     try std.testing.expect(result.error_info == null);
//     try std.testing.expect(result.node != null);
//     const node = result.node.?;
//     try std.testing.expect(node == .Scalar);
//     try std.testing.expectEqual(42.0, node.Scalar);
// }

// test "parse: addition" {
//     const tokens = comptime tokenize("1+2");
//     const result = comptime parse(tokens);
//     try std.testing.expect(result.error_info == null);
//     try std.testing.expect(result.node != null);
//     const node = result.node.?;
//     try std.testing.expect(node == .BinOp);
//     try std.testing.expectEqual(OpType.ADD, node.BinOp.op);
//     try std.testing.expectEqual(1.0, node.BinOp.lhs.Scalar);
//     try std.testing.expectEqual(2.0, node.BinOp.rhs.Scalar);
// }

// test "parse: multiplication" {
//     const tokens = comptime tokenize("2*3");
//     const result = comptime parse(tokens);
//     try std.testing.expect(result.error_info == null);
//     const node = result.node.?;
//     try std.testing.expect(node == .BinOp);
//     try std.testing.expectEqual(OpType.MUL, node.BinOp.op);
// }

// test "parse: precedence (multiplication before addition)" {
//     const tokens = comptime tokenize("1+2*3");
//     const result = comptime parse(tokens);
//     try std.testing.expect(result.error_info == null);
//     const node = result.node.?;
//     // Should parse as: 1 + (2 * 3)
//     try std.testing.expect(node == .BinOp);
//     try std.testing.expectEqual(OpType.ADD, node.BinOp.op);
//     try std.testing.expectEqual(1.0, node.BinOp.lhs.Scalar);
//     try std.testing.expect(node.BinOp.rhs.* == .BinOp);
//     try std.testing.expectEqual(OpType.MUL, node.BinOp.rhs.BinOp.op);
// }

// test "parse: parentheses override precedence" {
//     const tokens = comptime tokenize("(1+2)*3");
//     const result = comptime parse(tokens);
//     try std.testing.expect(result.error_info == null);
//     const node = result.node.?;
//     // Should parse as: (1 + 2) * 3
//     try std.testing.expect(node == .BinOp);
//     try std.testing.expectEqual(OpType.MUL, node.BinOp.op);
//     try std.testing.expect(node.BinOp.lhs.* == .BinOp);
//     try std.testing.expectEqual(OpType.ADD, node.BinOp.lhs.BinOp.op);
// }

// test "parse: error on invalid token" {
//     const tokens = comptime tokenize("1@2");
//     const result = comptime parse(tokens);
//     try std.testing.expect(result.error_info != null);
// }

test "tag parser: match" {
    const input_data = "hello world";
    const input = input_data;
    const result = tag("hello").parse(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("hello", result.?.match);
    try std.testing.expectEqualStrings(" world", result.?.rest);
}

test "tag parser: no match" {
    const input_data = "hello world";
    const input = input_data;
    const result = tag("goodbye").parse(input);
    try std.testing.expect(result == null);
}

test "opt combinator: successful parse" {
    const input_data = "   hello";
    const input = input_data;
    const maybe_ws = opt(whitespace);
    const result = maybe_ws.parse(input);
    try std.testing.expect(result != null);
    // Should have consumed the whitespace
    try std.testing.expectEqualStrings("   ", result.?.match.?);
    try std.testing.expectEqualStrings("hello", result.?.rest);
}

test "opt combinator: failed parse still succeeds" {
    const input_data = "hello";
    const input = input_data;
    const maybe_ws = opt(whitespace);
    const result = maybe_ws.parse(input);
    try std.testing.expect(result != null);
    // Should not have consumed anything (empty match)
    try std.testing.expectEqual(null, result.?.match);
    try std.testing.expectEqualStrings("hello", result.?.rest);
}

test "alt combinator: first parser matches" {
    const input_data = "hello world";
    const input = input_data;
    const parser = alt([]const u8, .{ tag("hello"), tag("goodbye") });
    const result = parser.parse(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("hello", result.?.match);
    try std.testing.expectEqualStrings(" world", result.?.rest);
}

test "alt combinator: second parser matches" {
    const input_data = "goodbye world";
    const input = input_data;
    const parser = alt([]const u8, .{ tag("hello"), tag("goodbye") });
    const result = parser.parse(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("goodbye", result.?.match);
    try std.testing.expectEqualStrings(" world", result.?.rest);
}

test "alt combinator: no match" {
    const input_data = "other world";
    const input = input_data;
    const parser = alt([]const u8, .{ tag("hello"), tag("goodbye") });
    const result = parser.parse(input);
    try std.testing.expect(result == null);
}

test "seq combinator: all parsers match" {
    const input_data = "hello world";
    const input = input_data;
    const parser = seq(.{ tag("hello"), whitespace, tag("world") });
    const result = parser.parse(input);
    try std.testing.expect(result != null);
    // Check tuple elements
    try std.testing.expectEqualStrings("hello", result.?.match[0]);
    try std.testing.expectEqualStrings(" ", result.?.match[1]);
    try std.testing.expectEqualStrings("world", result.?.match[2]);
    // Check rest is empty
    try std.testing.expectEqualStrings("", result.?.rest);
}

test "seq combinator: first parser fails" {
    const input_data = "goodbye world";
    const input = input_data;
    const parser = seq(.{ tag("hello"), whitespace, tag("world") });
    const result = parser.parse(input);
    try std.testing.expect(result == null);
}

test "seq combinator: middle parser fails" {
    const input_data = "helloworld";
    const input = input_data;
    const parser = seq(.{ tag("hello"), whitespace, tag("world") });
    const result = parser.parse(input);
    try std.testing.expect(result == null);
}

test "seq combinator: last parser fails" {
    const input_data = "hello goodbye";
    const input = input_data;
    const parser = seq(.{ tag("hello"), whitespace, tag("world") });
    const result = parser.parse(input);
    try std.testing.expect(result == null);
}

test "seq combinator: single parser" {
    const input_data = "hello";
    const input = input_data;
    const parser = seq(.{tag("hello")});
    const result = parser.parse(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("hello", result.?.match[0]);
}

test "recognize combinator: captures consumed input" {
    const input_data = "hello world";
    const input = input_data;
    const parser = recognize(seq(.{ tag("hello"), whitespace, tag("world") }));
    const result = parser.parse(input);
    try std.testing.expect(result != null);
    // Should return the entire matched input as a Range
    try std.testing.expectEqualStrings("hello world", result.?.match);
    try std.testing.expectEqualStrings("", result.?.rest);
}

test "recognize combinator: partial match" {
    const input_data = "hello foo";
    const input = input_data;
    const parser = recognize(tag("hello"));
    const result = parser.parse(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("hello", result.?.match);
    try std.testing.expectEqualStrings(" foo", result.?.rest);
}

test "recognize combinator: no match" {
    const input_data = "goodbye world";
    const input = input_data;
    const parser = recognize(tag("hello"));
    const result = parser.parse(input);
    try std.testing.expect(result == null);
}

test "number: positive integer" {
    const input_data = "123";
    const input = input_data;
    const result = number.parse(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("123", result.?.match);
}

test "number: negative integer" {
    const input_data = "-456";
    const input = input_data;
    const result = number.parse(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("-456", result.?.match);
}

test "number: decimal" {
    const input_data = "3.14";
    const input = input_data;
    const result = number.parse(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("3.14", result.?.match);
}

test "number: with exponent" {
    const input_data = "1e10";
    const input = input_data;
    const result = number.parse(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("1e10", result.?.match);
}

test "number: with negative exponent" {
    const input_data = "1e-5";
    const input = input_data;
    const result = number.parse(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("1e-5", result.?.match);
}

test "number: single digit" {
    const input_data = "5";
    const input = input_data;
    const result = number.parse(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("5", result.?.match);
}

test "number: partial match with remainder" {
    const input_data = "42 hello";
    const input = input_data;
    const result = number.parse(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("42", result.?.match);
    try std.testing.expectEqualStrings(" hello", result.?.rest);
}

test "number: not a number" {
    const input_data = "hello";
    const input = input_data;
    const result = number.parse(input);
    try std.testing.expect(result == null);
}

// Demonstrate compile-time parsing validation
fn isValidNumber(comptime str: []const u8) bool {
    const result = number.parse(str);
    return result != null;
}

test "comptime number validation" {
    // These checks happen entirely at compile time!
    comptime {
        if (!isValidNumber("123.45")) {
            @compileError("123.45 should be a valid number");
        }
        if (!isValidNumber("-42")) {
            @compileError("-42 should be a valid number");
        }
        if (!isValidNumber("1e-5")) {
            @compileError("1e-5 should be a valid number");
        }
        if (isValidNumber("abc")) {
            @compileError("abc should NOT be a valid number");
        }
        if (isValidNumber("")) {
            @compileError("empty string should NOT be a valid number");
        }
    }
}

test "map: parse number string to f32" {
    const parseFloat = struct {
        fn f(s: []const u8) f32 {
            return std.fmt.parseFloat(f32, s) catch 0.0;
        }
    }.f;

    const float_parser = map(parseFloat, number);
    const result = float_parser.parse("123.45 hello");

    try std.testing.expect(result != null);
    try std.testing.expectEqual(123.45, result.?.match);
    try std.testing.expectEqualStrings(" hello", result.?.rest);
}

test "map: transform string to length at compile time" {
    const getLength = struct {
        fn f(s: []const u8) usize {
            return s.len;
        }
    }.f;

    comptime {
        const length_parser = map(getLength, tag("hello"));
        if (length_parser.parse("hello world").?.match != 5) {
            @compileError("Expected a string length of 5");
        }
    }
}

test "value: identify a token" {
    const TestToken = enum { NUM };
    const result = value(TestToken.NUM, number).parse("123.45 hello");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(TestToken.NUM, result.?.match);
    try std.testing.expectEqualStrings(" hello", result.?.rest);
}

test "delimited: parenthesis aroudn a number" {
    const result = delimited(tag("("), number, tag(")")).parse("(42)more");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("42", result.?.match);
    try std.testing.expectEqualStrings("more", result.?.rest);
}

test "many: comma-separated numbers" {
    const parser = many(terminated(number, opt(tag(","))));
    const input_data = "1,2,3,4 rest";
    const input = input_data;
    comptime {
        const result = parser.parse(input);
        if (result == null) {
            @compileError("Expected successful parse");
        }
        if (result.?.match.len != 4) {
            @compileError("Expected 4 matches");
        }
    }
}

test "many: fails when no matches" {
    const parser = many(tag("hello"));
    const input_data = "goodbye";
    const input = input_data;
    comptime {
        const result = parser.parse(input);
        if (result != null) {
            @compileError("Expected parse to fail");
        }
    }
}
