const std = @import("std");

pub fn ParseResult(T: type, I: type) type {
    return struct {
        match: T,
        rest: I,
    };
}

/// Extract the match type from a parser type
fn ParserMatchType(comptime ParserType: type) type {
    const parse_return_type = @typeInfo(@TypeOf(ParserType.parse)).@"fn".return_type.?;
    const parse_result_type = @typeInfo(parse_return_type).optional.child;
    const fields = @typeInfo(parse_result_type).@"struct".fields;
    for (fields) |field| {
        if (std.mem.eql(u8, field.name, "match")) {
            return field.type;
        }
    }
    @compileError("Parser does not have a match field");
}

/// Extract the input type from a parser type
fn ParserInputType(comptime ParserType: type) type {
    const parse_return_type = @typeInfo(@TypeOf(ParserType.parse)).@"fn".return_type.?;
    const parse_result_type = @typeInfo(parse_return_type).optional.child;
    const fields = @typeInfo(parse_result_type).@"struct".fields;
    for (fields) |field| {
        if (std.mem.eql(u8, field.name, "rest")) {
            return field.type;
        }
    }
    @compileError("Parser does not have a rest field");
}

/// Extract return type from function or function pointer
fn FunctionReturnType(comptime func: anytype) type {
    const func_type_info = @typeInfo(@TypeOf(func));
    return switch (func_type_info) {
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
}

/// Extract input type from function's first parameter
fn FunctionInputType(comptime func: anytype) type {
    const func_type_info = @typeInfo(@TypeOf(func));
    const params = switch (func_type_info) {
        .@"fn" => |fn_info| fn_info.params,
        .pointer => |ptr_info| blk: {
            const pointee_info = @typeInfo(ptr_info.child);
            if (pointee_info == .@"fn") {
                break :blk pointee_info.@"fn".params;
            }
            @compileError("Expected function or function pointer");
        },
        else => @compileError("Expected function or function pointer"),
    };

    if (params.len == 0) {
        @compileError("Parser function must have at least one parameter");
    }

    const input_type = params[0].type.?;

    // Validate that it's a slice type []const T
    const input_info = @typeInfo(input_type);
    if (input_info != .pointer) {
        @compileError("First parameter must be a slice type []const T");
    }

    const ptr_info = input_info.pointer;
    if (ptr_info.size != .slice) {
        @compileError("First parameter must be a slice type []const T");
    }

    return input_type;
}

/// Validate that a type is ?ParseResult(T, I) and return the struct type
fn ValidateParseResult(comptime ReturnType: type) type {
    const type_info = @typeInfo(ReturnType);
    if (type_info != .optional) {
        @compileError("Parser function must return an optional type");
    }

    const ChildType = type_info.optional.child;
    const child_info = @typeInfo(ChildType);
    if (child_info != .@"struct") {
        @compileError("Parser result must be a ParseResult");
    }

    var has_match = false;
    var has_rest = false;
    for (child_info.@"struct".fields) |field| {
        if (std.mem.eql(u8, field.name, "match")) has_match = true;
        if (std.mem.eql(u8, field.name, "rest")) has_rest = true;
    }
    if (!has_match or !has_rest) {
        @compileError("Parser result must be a ParseResult");
    }

    return ChildType;
}

/// Creates a Parser type from a compile-time closure.
///
/// Contract:
/// - `args`: Any compile-time value to capture (e.g., a predicate function, a string, etc.)
/// - `func`: A function with signature `fn(InputType, @TypeOf(args)) ?ParseResult(T, InputType)` for some types T and InputType
///
/// The returned Parser type has a `parse(InputType)` method that returns `?ParseResult(T, InputType)`.
/// The InputType is automatically extracted from the function's first parameter.
///
/// Example:
///   ```zig
///   const my_parser = Parser("hello", struct {
///       fn parse(input: []const u8, target: []const u8) ?ParseResult([]const u8, []const u8) {
///           // ... parsing logic ...
///       }
///   }.parse);
///   ```
pub fn Parser(comptime args: anytype, comptime func: anytype) type {
    const ReturnType = FunctionReturnType(func);
    _ = ValidateParseResult(ReturnType);
    const InputType = FunctionInputType(func);

    return struct {
        const Self = @This();

        pub fn parse(input: InputType, comptime depth: u32) ReturnType {
            if (depth <= 1) {
                @compileError("Parser recursion depth limit reached. Consider increasing initial depth or simplifying grammar.");
            }
            return func(input, args, depth);
        }

        // Keep backward compatibility - default depth of 100
        pub fn parseSimple(input: InputType) ReturnType {
            return parse(input, 100);
        }
    };
}

/// Returns a Parser that matches while the predicate is true.
/// NOTE: Relies on Utf8Iterator.i pointing to the byte position
///       after the last consumed codepoint.
/// NOTE: zig decodes utf-8 codepoints to u21.
pub fn take_while(predicate: fn (u21) bool) type {
    return Parser(predicate, struct {
        fn parse(input: []const u8, predicate_: fn (u21) bool, comptime depth: u32) ?ParseResult([]const u8, []const u8) {
            _ = depth; // Not used for non-recursive parsers
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

pub fn take_one(predicate: fn (u21) bool) type {
    return Parser(predicate, struct {
        fn parse(input: []const u8, predicate_: fn (u21) bool, comptime depth: u32) ?ParseResult([]const u8, []const u8) {
            _ = depth; // Not used for non-recursive parsers
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

pub fn isWhitespace(codepoint: u21) bool {
    // This works because any non-ascii character will have the most
    // significant bit set in the low byte.
    return std.ascii.isWhitespace(@truncate(codepoint));
}

pub fn isDigit(codepoint: u21) bool {
    // This works because any non-ascii character will have the most
    // significant bit set in the low byte.
    return std.ascii.isDigit(@truncate(codepoint));
}

pub fn isAlpha(codepoint: u21) bool {
    // This works because any non-ascii character will have the most
    // significant bit set in the low byte.
    return std.ascii.isAlphabetic(@truncate(codepoint));
}

pub fn isAlphanumeric(codepoint: u21) bool {
    // This works because any non-ascii character will have the most
    // significant bit set in the low byte.
    return std.ascii.isAlphanumeric(@truncate(codepoint));
}

pub const whitespace = take_while(isWhitespace);
pub const whitespace0 = opt(whitespace);
pub const alphanumeric = take_while(isAlphanumeric);
pub const alphanumeric0 = opt(alphanumeric);
pub const digits = take_while(isDigit);

/// Returns a Parser that matches the given target string.
pub fn tag(target: []const u8) type {
    return Parser(target, struct {
        fn parse(input: []const u8, target_: []const u8, comptime depth: u32) ?ParseResult([]const u8, []const u8) {
            _ = depth; // Not used for non-recursive parsers
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
pub fn opt(comptime parser: type) type {
    const match_type = ParserMatchType(parser);
    const input_type = ParserInputType(parser);

    return Parser(parser, struct {
        fn parse(input: input_type, parser_: type, comptime depth: u32) ?ParseResult(?match_type, input_type) {
            if (parser_.parse(input, depth)) |result| {
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
/// The match type is automatically inferred from the first parser.
///
/// Example: `alt(.{ tag("hello"), tag("goodbye") })`
pub fn alt(comptime parsers: anytype) type {
    const parsers_type_info = @typeInfo(@TypeOf(parsers));

    // Convert tuple to array of types
    const parser_types = switch (parsers_type_info) {
        .@"struct" => |struct_info| blk: {
            if (!struct_info.is_tuple) {
                @compileError("Expected tuple of parser types, e.g., .{ tag(\"hello\"), tag(\"goodbye\") }");
            }
            if (struct_info.fields.len == 0) {
                @compileError("alt requires at least one parser");
            }
            comptime var types: [struct_info.fields.len]type = undefined;
            inline for (struct_info.fields, 0..) |field, i| {
                types[i] = @field(parsers, field.name);
            }
            break :blk types;
        },
        .pointer => |ptr_info| blk: {
            if (ptr_info.size == .slice and ptr_info.child == type) {
                // Also accept slice syntax for backwards compatibility
                if (parsers.len == 0) {
                    @compileError("alt requires at least one parser");
                }
                break :blk parsers;
            }
            @compileError("Expected tuple of parser types");
        },
        else => @compileError("Expected tuple of parser types, e.g., .{ tag(\"hello\"), tag(\"goodbye\") }"),
    };

    // Extract the match type from the first parser and validate all have the same type
    const FirstMatchType = ParserMatchType(parser_types[0]);
    const FirstInputType = ParserInputType(parser_types[0]);
    comptime {
        for (parser_types[1..], 1..) |ParserType, i| {
            const MatchType = ParserMatchType(ParserType);
            const InputType = ParserInputType(ParserType);
            if (MatchType != FirstMatchType) {
                @compileError(std.fmt.comptimePrint("Parser {} has match type {s}, but expected {s} (from first parser)", .{ i, @typeName(MatchType), @typeName(FirstMatchType) }));
            }
            if (InputType != FirstInputType) {
                @compileError(std.fmt.comptimePrint("Parser {} has input type {s}, but expected {s} (from first parser)", .{ i, @typeName(InputType), @typeName(FirstInputType) }));
            }
        }
    }

    return Parser(parser_types, struct {
        fn parse(input: FirstInputType, ps: @TypeOf(parser_types), comptime depth: u32) ?ParseResult(FirstMatchType, FirstInputType) {
            inline for (ps) |ParserType| {
                if (ParserType.parse(input, depth)) |result| {
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
pub fn seq(comptime parsers: anytype) type {
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
            if (ptr_info.size == .slice and ptr_info.child == type) {
                // Also accept slice syntax for backwards compatibility
                break :blk parsers;
            }
            @compileError("Expected tuple of Parser types");
        },
        else => @compileError("Expected tuple of Parser types, e.g., .{ tag(\"hello\"), whitespace }"),
    };

    // Extract the match type from each parser and validate all have the same input type
    comptime var match_types: [parser_types.len]type = undefined;
    const FirstInputType = ParserInputType(parser_types[0]);
    inline for (parser_types, 0..) |ParserType, i| {
        match_types[i] = ParserMatchType(ParserType);
        const InputType = ParserInputType(ParserType);
        if (InputType != FirstInputType) {
            @compileError(std.fmt.comptimePrint("Parser {} has input type {s}, but expected {s} (from first parser)", .{ i, @typeName(InputType), @typeName(FirstInputType) }));
        }
    }

    // Create a tuple type from all the match types
    const TupleType = std.meta.Tuple(&match_types);

    return Parser(parser_types, struct {
        fn parse(input: FirstInputType, ps: @TypeOf(parser_types), comptime depth: u32) ?ParseResult(TupleType, FirstInputType) {
            var current = input;
            var results: TupleType = undefined;

            inline for (ps, 0..) |ParserType, i| {
                const result = ParserType.parse(current, depth) orelse return null;
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
/// Note: This is string-specific and requires []const u8 input.
pub fn recognize(comptime parser: type) type {
    const input_type = ParserInputType(parser);
    if (input_type != []const u8) {
        @compileError("recognize combinator only works with []const u8 input types");
    }

    return Parser(parser, struct {
        fn parse(input: []const u8, parser_: type, comptime depth: u32) ?ParseResult([]const u8, []const u8) {
            const result = parser_.parse(input, depth) orelse return null;
            const consumed = input.len - result.rest.len;
            return .{
                .match = input[0..consumed],
                .rest = result.rest,
            };
        }
    }.parse);
}

/// Matches numbers e.g -123.45e-67
pub const number = recognize(seq(.{
    opt(tag("-")), // optional sign
    alt(.{ tag("0"), digits }), // integer part ('0' or 1+ digits)
    opt(seq(.{ tag("."), digits })), // decimal part
    opt(seq(.{ tag("e"), opt(tag("-")), digits })), // optional exponent
}));

/// Maps the match type for the parser
pub fn map(comptime mapfn: anytype, comptime parser: type) type {
    // 1. Extract the match type from the parser
    const match_type = ParserMatchType(parser);
    const input_type = ParserInputType(parser);

    // 2. Validate and extract mapfn information
    const mapfn_info = @typeInfo(@TypeOf(mapfn));
    if (mapfn_info != .@"fn") {
        @compileError("map expects a function");
    }

    const params = mapfn_info.@"fn".params;
    if (params.len != 1) {
        @compileError("map function must have exactly 1 parameter");
    }

    // Only check parameter type if it's available (not anytype)
    if (params[0].type) |param_type| {
        if (param_type != match_type) {
            @compileError("map function parameter type must match parser match type");
        }
    }

    const MappedType = mapfn_info.@"fn".return_type.?;

    // 3. Create and return the mapped parser
    return Parser(.{ .parser = parser, .mapfn = mapfn }, struct {
        fn parse(input: input_type, ctx: anytype, comptime depth: u32) ?ParseResult(MappedType, input_type) {
            const result = ctx.parser.parse(input, depth) orelse return null;
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
pub fn many(comptime parser: type) type {
    const match_type = ParserMatchType(parser);
    const input_type = ParserInputType(parser);

    return Parser(parser, struct {
        fn parse(input: input_type, parser_: type, comptime depth: u32) ?ParseResult([]const match_type, input_type) {
            var results: []const match_type = &.{};
            var current = input;
            var iterations: usize = 0;
            const max_iterations = 1000;

            while (iterations < max_iterations) : (iterations += 1) {
                if (parser_.parse(current, depth)) |result| {
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

/// Create a lazy parser that defers type instantiation to break circular dependencies
/// The parser_fn should return a parser type when called
pub fn lazy(comptime parser_fn: fn () type) type {
    // We need to get the types somehow - let's call the function once to get them
    // This works because the circular dependency only matters when we try to instantiate
    // all the types at once
    const DummyParser = parser_fn();
    const match_type = ParserMatchType(DummyParser);
    const input_type = ParserInputType(DummyParser);

    return Parser(parser_fn, struct {
        fn parse(input: input_type, parser_getter: fn () type, comptime depth: u32) ?ParseResult(match_type, input_type) {
            const parser = parser_getter();
            return parser.parse(input, depth);
        }
    }.parse);
}

/// Creates a recursive parser that decrements depth on each recursion.
/// The parser_fn should be a function that takes depth and returns a parser type.
/// Example: recursive(100, struct { fn p(comptime d: u32) type { return expr_parser(d-1); } }.p)
pub fn recursive(comptime parser_fn: fn (comptime u32) type) type {
    // Get the parser type for depth 100 to extract types
    const SampleParser = parser_fn(100);
    const match_type = ParserMatchType(SampleParser);
    const input_type = ParserInputType(SampleParser);

    return Parser(parser_fn, struct {
        fn parse(input: input_type, parser_fn_: fn (comptime u32) type, comptime depth: u32) ?ParseResult(match_type, input_type) {
            if (depth == 0) {
                return null; // Maximum recursion depth reached
            }
            const parser = parser_fn_(depth - 1);
            return parser.parse(input, depth - 1);
        }
    }.parse);
}

/// Fold a parser's results with an accumulator function.
/// Parses one or more times and accumulates results from left to right.
/// accumulator_fn should have signature: fn(acc: AccType, item: MatchType) AccType
pub fn fold(comptime parser: type, comptime initial: anytype, comptime accumulator_fn: anytype) type {
    const input_type = ParserInputType(parser);
    const AccType = @TypeOf(initial);

    return Parser(.{ .parser = parser, .initial = initial, .accumulator = accumulator_fn }, struct {
        fn parse(input: input_type, ctx: anytype, comptime depth: u32) ?ParseResult(AccType, input_type) {
            var current = input;
            var acc = ctx.initial;
            var has_match = false;

            // Need at least one match
            while (ctx.parser.parse(current, depth)) |result| {
                acc = ctx.accumulator(acc, result.match);
                current = result.rest;
                has_match = true;
            }

            if (!has_match) {
                return null;
            }

            return .{
                .match = acc,
                .rest = current,
            };
        }
    }.parse);
}

/// Returns the provided value if the child parser succeeds.
pub fn value(comptime val: anytype, comptime parser: type) type {
    const input_type = ParserInputType(parser);

    return Parser(.{ .parser = parser, .value = val }, struct {
        fn parse(input: input_type, ctx: anytype, comptime depth: u32) ?ParseResult(@TypeOf(val), input_type) {
            const result = ctx.parser.parse(input, depth) orelse return null;
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
pub fn delimited(comptime FirstParser: type, comptime SecondParser: type, comptime ThirdParser: type) type {
    const match_type = ParserMatchType(SecondParser);
    const input_type = ParserInputType(SecondParser);

    return Parser(.{ FirstParser, SecondParser, ThirdParser }, struct {
        fn parse(input: input_type, ctx: anytype, comptime depth: u32) ?ParseResult(match_type, input_type) {
            const result = seq(ctx).parse(input, depth) orelse return null;
            return .{
                .match = result.match[1],
                .rest = result.rest,
            };
        }
    }.parse);
}

/// Gets an object from the Value parser, and matches an object from the
/// Terminator parser and discards it.
pub fn terminated(comptime ValueParser: type, comptime TerminatorParser: type) type {
    const match_type = ParserMatchType(ValueParser);
    const input_type = ParserInputType(ValueParser);

    return Parser(.{ ValueParser, TerminatorParser }, struct {
        fn parse(input: input_type, ctx: anytype, comptime depth: u32) ?ParseResult(match_type, input_type) {
            const result = seq(ctx).parse(input, depth) orelse return null;
            return .{
                .match = result.match[0],
                .rest = result.rest,
            };
        }
    }.parse);
}

/// Gets an object from the Value parser, and matches an object from the
/// Terminator parser and discards it.
pub fn preceded(comptime FirstParser: type, comptime ValueParser: type) type {
    const match_type = ParserMatchType(ValueParser);
    const input_type = ParserInputType(ValueParser);

    return Parser(.{ FirstParser, ValueParser }, struct {
        fn parse(input: input_type, ctx: anytype, comptime depth: u32) ?ParseResult(match_type, input_type) {
            const result = seq(ctx).parse(input, depth) orelse return null;
            return .{
                .match = result.match[0],
                .rest = result.rest,
            };
        }
    }.parse);
}
test "tag parser: match" {
    const input_data = "hello world";
    const input = input_data;
    const result = tag("hello").parseSimple(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("hello", result.?.match);
    try std.testing.expectEqualStrings(" world", result.?.rest);
}

test "tag parser: no match" {
    const input_data = "hello world";
    const input = input_data;
    const result = tag("goodbye").parseSimple(input);
    try std.testing.expect(result == null);
}

test "opt combinator: successful parse" {
    const input_data = "   hello";
    const input = input_data;
    const maybe_ws = opt(whitespace);
    const result = maybe_ws.parseSimple(input);
    try std.testing.expect(result != null);
    // Should have consumed the whitespace
    try std.testing.expectEqualStrings("   ", result.?.match.?);
    try std.testing.expectEqualStrings("hello", result.?.rest);
}

test "opt combinator: failed parse still succeeds" {
    const input_data = "hello";
    const input = input_data;
    const maybe_ws = opt(whitespace);
    const result = maybe_ws.parseSimple(input);
    try std.testing.expect(result != null);
    // Should not have consumed anything (empty match)
    try std.testing.expectEqual(null, result.?.match);
    try std.testing.expectEqualStrings("hello", result.?.rest);
}

test "alt combinator: first parser matches" {
    const input_data = "hello world";
    const input = input_data;
    const parser = alt(.{ tag("hello"), tag("goodbye") });
    const result = parser.parseSimple(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("hello", result.?.match);
    try std.testing.expectEqualStrings(" world", result.?.rest);
}

test "alt combinator: second parser matches" {
    const input_data = "goodbye world";
    const input = input_data;
    const parser = alt(.{ tag("hello"), tag("goodbye") });
    const result = parser.parseSimple(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("goodbye", result.?.match);
    try std.testing.expectEqualStrings(" world", result.?.rest);
}

test "alt combinator: no match" {
    const input_data = "other world";
    const input = input_data;
    const parser = alt(.{ tag("hello"), tag("goodbye") });
    const result = parser.parseSimple(input);
    try std.testing.expect(result == null);
}

test "seq combinator: all parsers match" {
    const input_data = "hello world";
    const input = input_data;
    const parser = seq(.{ tag("hello"), whitespace, tag("world") });
    const result = parser.parseSimple(input);
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
    const result = parser.parseSimple(input);
    try std.testing.expect(result == null);
}

test "seq combinator: middle parser fails" {
    const input_data = "helloworld";
    const input = input_data;
    const parser = seq(.{ tag("hello"), whitespace, tag("world") });
    const result = parser.parseSimple(input);
    try std.testing.expect(result == null);
}

test "seq combinator: last parser fails" {
    const input_data = "hello goodbye";
    const input = input_data;
    const parser = seq(.{ tag("hello"), whitespace, tag("world") });
    const result = parser.parseSimple(input);
    try std.testing.expect(result == null);
}

test "seq combinator: single parser" {
    const input_data = "hello";
    const input = input_data;
    const parser = seq(.{tag("hello")});
    const result = parser.parseSimple(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("hello", result.?.match[0]);
}

test "recognize combinator: captures consumed input" {
    const input_data = "hello world";
    const input = input_data;
    const parser = recognize(seq(.{ tag("hello"), whitespace, tag("world") }));
    const result = parser.parseSimple(input);
    try std.testing.expect(result != null);
    // Should return the entire matched input as a Range
    try std.testing.expectEqualStrings("hello world", result.?.match);
    try std.testing.expectEqualStrings("", result.?.rest);
}

test "recognize combinator: partial match" {
    const input_data = "hello foo";
    const input = input_data;
    const parser = recognize(tag("hello"));
    const result = parser.parseSimple(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("hello", result.?.match);
    try std.testing.expectEqualStrings(" foo", result.?.rest);
}

test "recognize combinator: no match" {
    const input_data = "goodbye world";
    const input = input_data;
    const parser = recognize(tag("hello"));
    const result = parser.parseSimple(input);
    try std.testing.expect(result == null);
}

test "number: positive integer" {
    const input_data = "123";
    const input = input_data;
    const result = number.parseSimple(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("123", result.?.match);
}

test "number: negative integer" {
    const input_data = "-456";
    const input = input_data;
    const result = number.parseSimple(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("-456", result.?.match);
}

test "number: decimal" {
    const input_data = "3.14";
    const input = input_data;
    const result = number.parseSimple(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("3.14", result.?.match);
}

test "number: with exponent" {
    const input_data = "1e10";
    const input = input_data;
    const result = number.parseSimple(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("1e10", result.?.match);
}

test "number: with negative exponent" {
    const input_data = "1e-5";
    const input = input_data;
    const result = number.parseSimple(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("1e-5", result.?.match);
}

test "number: single digit" {
    const input_data = "5";
    const input = input_data;
    const result = number.parseSimple(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("5", result.?.match);
}

test "number: partial match with remainder" {
    const input_data = "42 hello";
    const input = input_data;
    const result = number.parseSimple(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("42", result.?.match);
    try std.testing.expectEqualStrings(" hello", result.?.rest);
}

test "number: not a number" {
    const input_data = "hello";
    const input = input_data;
    const result = number.parseSimple(input);
    try std.testing.expect(result == null);
}

// Demonstrate compile-time parsing validation
fn isValidNumber(comptime str: []const u8) bool {
    const result = number.parseSimple(str);
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
    const result = float_parser.parseSimple("123.45 hello");

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
        if (length_parser.parseSimple("hello world").?.match != 5) {
            @compileError("Expected a string length of 5");
        }
    }
}

test "value: identify a token" {
    const TestToken = enum { NUM };
    const result = value(TestToken.NUM, number).parseSimple("123.45 hello");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(TestToken.NUM, result.?.match);
    try std.testing.expectEqualStrings(" hello", result.?.rest);
}

test "delimited: parenthesis around a number" {
    const result = delimited(tag("("), number, tag(")")).parseSimple("(42)more");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("42", result.?.match);
    try std.testing.expectEqualStrings("more", result.?.rest);
}

test "many: comma-separated numbers" {
    const parser = many(terminated(number, opt(tag(","))));
    const input_data = "1,2,3,4 rest";
    const input = input_data;
    comptime {
        const result = parser.parseSimple(input);
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
        const result = parser.parseSimple(input);
        if (result != null) {
            @compileError("Expected parse to fail");
        }
    }
}

test "alt: automatic type inference from first parser" {
    // Test that alt() now works without explicit type parameter
    const parser = alt(.{ tag("foo"), tag("bar") });
    const result = parser.parseSimple("foo baz");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("foo", result.?.match);
    try std.testing.expectEqualStrings(" baz", result.?.rest);
}
