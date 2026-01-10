const std = @import("std");
const P = @import("parse.zig");

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

const valid_identifier = P.recognize(P.seq(.{ P.take_one(P.isAlpha), P.alphanumeric0 }));

// Number parser without negative sign (we want minus to be a separate token)
const unsigned_number = P.recognize(P.seq(.{
    P.alt(.{ P.tag("0"), P.digits }), // integer part ('0' or 1+ digits)
    P.opt(P.seq(.{ P.tag("."), P.digits })), // decimal part
    P.opt(P.seq(.{ P.tag("e"), P.opt(P.tag("-")), P.digits })), // optional exponent
}));

const tokenize = P.many(P.delimited(P.whitespace0, P.alt(.{
    // Operators should be checked before numbers
    P.map(mk_token(TokenType.PLUS), P.tag("+")),
    P.map(mk_token(TokenType.MINUS), P.tag("-")),
    P.map(mk_token(TokenType.STAR), P.tag("*")),
    P.map(mk_token(TokenType.LPAREN), P.tag("(")),
    P.map(mk_token(TokenType.RPAREN), P.tag(")")),
    // Then numbers and identifiers
    P.map(mk_token(TokenType.NUMBER), unsigned_number),
    P.map(mk_token(TokenType.IDENTIFIER), valid_identifier),
    // Invalid tokens
    // Try to consume any non-alphanumeric characters first.
    // If that doesn't work then just take everything up to
    // the next whitespace
    P.map(mk_token(TokenType.INVALID), P.take_while(struct {
        fn inner(codepoint: u21) bool {
            return !(P.isAlphanumeric(codepoint) or P.isWhitespace(codepoint));
        }
    }.inner)),
    P.map(mk_token(TokenType.INVALID), P.take_while(struct {
        fn inner(codepoint: u21) bool {
            return !P.isWhitespace(codepoint);
        }
    }.inner)),
}), P.whitespace0));

// === Token Parsers ===
// These parsers work with []const Token input

/// Creates a parser that matches a specific token type
pub fn token_tag(comptime token_type: TokenType) type {
    return P.Parser(token_type, struct {
        fn parse(input: []const Token, target_type: TokenType, comptime depth: u32) ?P.ParseResult(Token, []const Token) {
            _ = depth; // Not used for non-recursive parsers
            if (input.len == 0) return null;
            if (input[0].type == target_type) {
                return .{ .match = input[0], .rest = input[1..] };
            }
            return null;
        }
    }.parse);
}

/// Creates a parser that matches multiple token types
pub fn token_alt(comptime token_types: []const TokenType) type {
    return P.Parser(token_types, struct {
        fn parse(input: []const Token, types: []const TokenType, comptime depth: u32) ?P.ParseResult(Token, []const Token) {
            _ = depth; // Not used for non-recursive parsers
            if (input.len == 0) return null;
            for (types) |token_type| {
                if (input[0].type == token_type) {
                    return .{ .match = input[0], .rest = input[1..] };
                }
            }
            return null;
        }
    }.parse);
}

/// Creates a parser that matches tokens based on a predicate
pub fn token_where(predicate: fn (Token) bool) type {
    return P.Parser(predicate, struct {
        fn parse(input: []const Token, pred: fn (Token) bool, comptime depth: u32) ?P.ParseResult(Token, []const Token) {
            _ = depth; // Not used for non-recursive parsers
            if (input.len == 0) return null;
            if (pred(input[0])) {
                return .{ .match = input[0], .rest = input[1..] };
            }
            return null;
        }
    }.parse);
}

pub const OpType = enum {
    ADD,
    SUB,
    MUL,
};

/// Represents a span in the original source text
pub const SourceSpan = struct {
    text: []const u8, // The original source text for this span

    pub fn format(self: SourceSpan, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("\"{s}\"", .{self.text});
    }
};

pub const ParseErrorKind = enum {
    InvalidNumber,
    UnexpectedToken,
    MissingOperand,
    MissingClosingParen,
    InvalidExpression,
};

// Array-based AST node - no pointers needed
// For binary tree layout: node at index 0 is root,
// left child of node i is at 2i+1, right child at 2i+2
pub const ASTNode = struct {
    span: SourceSpan,
    kind: union(enum) {
        Scalar: f32,
        BinOp: OpType, // Just the operator, children are implicit in array layout
        Identifier: []const u8,
        Error: struct {
            kind: ParseErrorKind,
            message: []const u8,
        },
    },

    pub fn format(self: ASTNode, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        switch (self.kind) {
            .Scalar => |s| try writer.print("Scalar({d})", .{s}),
            .BinOp => |op| {
                const op_str = switch (op) {
                    .ADD => "+",
                    .SUB => "-",
                    .MUL => "*",
                };
                try writer.print("BinOp({s})", .{op_str});
            },
            .Identifier => |name| try writer.print("Identifier({s})", .{name}),
            .Error => |e| try writer.print("Error({s}: {s})", .{ @tagName(e.kind), e.message }),
        }
    }
};

fn mk_node_from_token(token: Token) ASTNode {
    return switch (token.type) {
        .NUMBER => blk: {
            if (std.fmt.parseFloat(f32, token.lexeme)) |val| {
                break :blk ASTNode{
                    .span = .{ .text = token.lexeme },
                    .kind = .{ .Scalar = val },
                };
            } else |_| {
                break :blk ASTNode{
                    .span = .{ .text = token.lexeme },
                    .kind = .{ .Error = .{
                        .kind = .InvalidNumber,
                        .message = "Failed to parse number",
                    } },
                };
            }
        },
        .IDENTIFIER => ASTNode{
            .span = .{ .text = token.lexeme },
            .kind = .{ .Identifier = token.lexeme },
        },
        else => ASTNode{
            .span = .{ .text = token.lexeme },
            .kind = .{ .Error = .{
                .kind = .UnexpectedToken,
                .message = "Unexpected token type",
            } },
        },
    };
}

const number = P.map(mk_node_from_token, token_tag(TokenType.NUMBER));
const plus = token_tag(TokenType.PLUS);
const minus = token_tag(TokenType.MINUS);
const star = token_tag(TokenType.STAR);
const lparen = token_tag(TokenType.LPAREN);
const rparen = token_tag(TokenType.RPAREN);
const identifier = token_tag(TokenType.IDENTIFIER);

// === Expression Parser using Combinators ===
// Grammar:
//   expr   -> term (('+' | '-') term)*
//   term   -> factor ('*' factor)*
//   factor -> NUMBER | IDENTIFIER | '(' expr ')'

// Helper function to build a binary operation node in array format
fn buildBinOp(op: OpType, left: []const ASTNode, right: []const ASTNode) []const ASTNode {
    const binop = ASTNode{
        .span = .{ .text = "" }, // Will be filled in later
        .kind = .{ .BinOp = op },
    };
    return &[_]ASTNode{binop} ++ left ++ right;
}

// Convert a single ASTNode to an array
fn nodeToArray(node: ASTNode) []const ASTNode {
    return &[_]ASTNode{node};
}

// Map parsers to produce []const ASTNode
const number_ast = P.map(nodeToArray, number);
const identifier_ast = P.map(nodeToArray, P.map(mk_node_from_token, identifier));

// We need depth parameters to break circular compile-time dependencies
// But we can make the API cleaner by having a wrapper

fn expr_parser_impl(comptime depth: u32) type {
    const safe_depth = if (depth > 0) depth else 1;
    const term = term_parser_impl(safe_depth - 1);
    const add_op = P.value(OpType.ADD, plus);
    const sub_op = P.value(OpType.SUB, minus);
    const add_or_sub = P.alt(.{ add_op, sub_op });

    // expr -> term (('+' | '-') term)*
    return P.map(
        struct {
            fn build(parts: anytype) []const ASTNode {
                const first_term = parts[0];
                const op_terms = parts[1]; // []const .{OpType, term}

                if (op_terms == null) {
                    return first_term;
                }

                // Left-associate the additions/subtractions
                var result = first_term;
                for (op_terms.?) |op_term| {
                    const op = op_term[0]; // OpType
                    const right = op_term[1]; // term
                    result = buildBinOp(op, result, right);
                }
                return result;
            }
        }.build,
        P.seq(.{
            term,
            P.opt(P.many(P.seq(.{ add_or_sub, term }))),
        })
    );
}

fn term_parser_impl(comptime depth: u32) type {
    const safe_depth = if (depth > 0) depth else 1;
    const factor = factor_parser_impl(safe_depth - 1);

    // term -> factor ('*' factor)*
    return P.map(
        struct {
            fn build(parts: anytype) []const ASTNode {
                const first_factor = parts[0];
                const mul_factors = parts[1]; // []const .{star_token, factor}

                if (mul_factors == null) {
                    return first_factor;
                }

                // Left-associate the multiplications
                var result = first_factor;
                for (mul_factors.?) |mul_pair| {
                    _ = mul_pair[0]; // star token
                    const right = mul_pair[1]; // factor
                    result = buildBinOp(OpType.MUL, result, right);
                }
                return result;
            }
        }.build,
        P.seq(.{
            factor,
            P.opt(P.many(P.seq(.{ star, factor }))),
        })
    );
}

fn factor_parser_impl(comptime depth: u32) type {
    const safe_depth = if (depth > 0) depth else 1;

    // factor -> NUMBER | IDENTIFIER | '(' expr ')'
    return P.alt(.{
        number_ast,
        identifier_ast,
        P.delimited(lparen, expr_parser_impl(safe_depth - 1), rparen),
    });
}

// Clean public API - just call with the initial depth from parseExpressionCombinatorWithDepth
fn expr_parser(comptime depth: u32) type {
    return expr_parser_impl(depth);
}

/// Parse expression using combinators (new version)
pub fn parseExpressionCombinator(tokens: []const Token, original_input: []const u8) []const ASTNode {
    _ = original_input; // Not used in combinator version yet
    return parseExpressionCombinatorWithDepth(tokens, 10); // Reduced depth for testing
}

/// Parse expression using combinators with configurable depth
pub fn parseExpressionCombinatorWithDepth(tokens: []const Token, comptime depth: u32) []const ASTNode {
    const parser = expr_parser(depth);
    const result = parser.parse(tokens, depth);

    if (result) |res| {
        // Check if we consumed all tokens
        if (res.rest.len > 0) {
            return &[_]ASTNode{ASTNode{
                .span = .{ .text = if (res.rest.len > 0) res.rest[0].lexeme else "" },
                .kind = .{ .Error = .{
                    .kind = .InvalidExpression,
                    .message = "Unexpected tokens after expression",
                } },
            }};
        }
        return res.match;
    } else {
        return &[_]ASTNode{ASTNode{
            .span = .{ .text = "" },
            .kind = .{ .Error = .{
                .kind = .InvalidExpression,
                .message = "Failed to parse expression",
            } },
        }};
    }
}

// === Old Recursive Descent Parser (kept for now) ===
// Traditional recursive descent parser for arithmetic expressions

const ParseState = struct {
    tokens: []const Token,
    pos: usize,
    original_input: []const u8, // Keep track of the original input string

    fn current(self: *const ParseState) ?Token {
        if (self.pos < self.tokens.len) {
            return self.tokens[self.pos];
        }
        return null;
    }

    fn advance(self: *ParseState) ?Token {
        if (self.pos < self.tokens.len) {
            const tok = self.tokens[self.pos];
            self.pos += 1;
            return tok;
        }
        return null;
    }

    fn peek(self: *const ParseState) ?Token {
        if (self.pos + 1 < self.tokens.len) {
            return self.tokens[self.pos + 1];
        }
        return null;
    }

    fn makeError(self: *const ParseState, kind: ParseErrorKind, message: []const u8) ASTNode {
        const span_text = if (self.current()) |tok| tok.lexeme else "EOF";
        return ASTNode{
            .span = .{ .text = span_text },
            .kind = .{ .Error = .{
                .kind = kind,
                .message = message,
            } },
        };
    }

    // Helper to get combined span from start token to current position
    fn getSpan(self: *const ParseState, start_pos: usize) []const u8 {
        // For now, just return the original input as the span
        // This avoids complex pointer arithmetic at compile time
        _ = start_pos;
        return self.original_input;
    }
};

/// Parse an expression: term (('+' | '-') term)*
fn parseExpr(state: *ParseState, comptime depth: u32) []const ASTNode {
    if (depth == 0) {
        return &[_]ASTNode{state.makeError(.InvalidExpression, "Expression too deeply nested")};
    }

    const start_pos = state.pos;
    var left = parseTerm(state, depth - 1);

    // Check if left is an error
    if (left.len == 1 and left[0].kind == .Error) {
        return left;
    }

    while (state.current()) |tok| {
        const op = switch (tok.type) {
            .PLUS => OpType.ADD,
            .MINUS => OpType.SUB,
            else => break,
        };

        _ = state.advance(); // consume operator
        const right = parseTerm(state, depth - 1);

        // Check if right is an error
        if (right.len == 1 and right[0].kind == .Error) {
            return right;
        }

        // Create binary op node as root with left and right as children
        const binop = ASTNode{
            .span = .{ .text = state.getSpan(start_pos) },
            .kind = .{ .BinOp = op },
        };

        // Concatenate: [root] ++ left_tree ++ right_tree
        left = &[_]ASTNode{binop} ++ left ++ right;
    }

    return left;
}

/// Parse a term: factor ('*' factor)*
fn parseTerm(state: *ParseState, comptime depth: u32) []const ASTNode {
    if (depth == 0) {
        return &[_]ASTNode{state.makeError(.InvalidExpression, "Expression too deeply nested")};
    }

    const start_pos = state.pos;
    var left = parseFactor(state, depth - 1);

    // Check if left is an error
    if (left.len == 1 and left[0].kind == .Error) {
        return left;
    }

    while (state.current()) |tok| {
        if (tok.type != .STAR) break;

        _ = state.advance(); // consume operator
        const right = parseFactor(state, depth - 1);

        // Check if right is an error
        if (right.len == 1 and right[0].kind == .Error) {
            return right;
        }

        // Create binary op node as root with left and right as children
        const binop = ASTNode{
            .span = .{ .text = state.getSpan(start_pos) },
            .kind = .{ .BinOp = OpType.MUL },
        };

        // Concatenate: [root] ++ left_tree ++ right_tree
        left = &[_]ASTNode{binop} ++ left ++ right;
    }

    return left;
}

/// Parse a factor: NUMBER | IDENTIFIER | '(' expr ')'
fn parseFactor(state: *ParseState, comptime depth: u32) []const ASTNode {
    if (depth == 0) {
        return &[_]ASTNode{state.makeError(.InvalidExpression, "Expression too deeply nested")};
    }

    const tok = state.current() orelse {
        return &[_]ASTNode{state.makeError(.MissingOperand, "Expected a number, identifier, or expression")};
    };

    switch (tok.type) {
        .NUMBER => {
            _ = state.advance();
            return &[_]ASTNode{mk_node_from_token(tok)};
        },
        .IDENTIFIER => {
            _ = state.advance();
            return &[_]ASTNode{mk_node_from_token(tok)};
        },
        .LPAREN => {
            _ = state.advance(); // consume '('
            const inner = parseExpr(state, depth - 1);

            // Check for error in inner expression
            if (inner.len == 1 and inner[0].kind == .Error) {
                return inner;
            }

            // Expect closing paren
            const close = state.current() orelse {
                return &[_]ASTNode{state.makeError(.MissingClosingParen, "Expected ')'")};
            };

            if (close.type != .RPAREN) {
                return &[_]ASTNode{state.makeError(.MissingClosingParen, "Expected ')'")};
            }

            _ = state.advance(); // consume ')'

            // Return the inner expression tree as-is
            return inner;
        },
        else => {
            return &[_]ASTNode{state.makeError(.UnexpectedToken, "Expected a number, identifier, or '('")};
        },
    }
}

/// Main parse function returning array of nodes
pub fn parseExpression(tokens: []const Token, original_input: []const u8) []const ASTNode {
    return parseExpressionWithDepth(tokens, original_input, 100);
}

/// Main parse function with configurable depth
pub fn parseExpressionWithDepth(tokens: []const Token, original_input: []const u8, comptime depth: u32) []const ASTNode {
    var state = ParseState{
        .tokens = tokens,
        .pos = 0,
        .original_input = original_input,
    };

    const result = parseExpr(&state, depth);

    // Check if we consumed all tokens
    if (state.pos < state.tokens.len) {
        return &[_]ASTNode{state.makeError(.InvalidExpression, "Unexpected tokens after expression")};
    }

    return result;
}

// ============================================================================
// Evaluator
// ============================================================================

/// Evaluate an AST from array representation
/// For binary tree: left child of node at index i is at 2i+1, right at 2i+2
fn evaluateAST(comptime BasisType: type, comptime nodes: []const ASTNode, comptime idx: usize, vars: anytype) BasisType.Multivector {
    if (idx >= nodes.len) {
        @compileError("Invalid node index");
    }

    const node = nodes[idx];
    switch (node.kind) {
        .Scalar => |val| {
            // Value is already parsed by the parser
            var result = BasisType.Multivector.zero();
            result.coefficients[0] = val;
            return result;
        },
        .BinOp => |op| {
            // For array layout: left child at idx+1, right child follows left subtree
            // We need to calculate where right subtree starts
            const left_idx = idx + 1;
            const left_size = getSubtreeSize(nodes, left_idx);
            const right_idx = left_idx + left_size;

            const lhs = evaluateAST(BasisType, nodes, left_idx, vars);
            const rhs = evaluateAST(BasisType, nodes, right_idx, vars);

            return switch (op) {
                .ADD => lhs.add(rhs),
                .SUB => lhs.sub(rhs),
                .MUL => lhs.mul(rhs),
            };
        },
        .Identifier => |_| {
            // TODO: implement variable lookup
            @compileError("Variable lookup not yet implemented");
        },
        .Error => |err| {
            @compileError("Parse error: " ++ err.message);
        },
    }
}

/// Get the size of a subtree (number of nodes)
fn getSubtreeSize(nodes: []const ASTNode, idx: usize) usize {
    if (idx >= nodes.len) return 0;

    const node = nodes[idx];
    switch (node.kind) {
        .Scalar, .Identifier, .Error => return 1,
        .BinOp => {
            // Size is 1 (root) + left subtree + right subtree
            const left_idx = idx + 1;
            const left_size = getSubtreeSize(nodes, left_idx);
            const right_idx = left_idx + left_size;
            const right_size = getSubtreeSize(nodes, right_idx);
            return 1 + left_size + right_size;
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
    const tokens = comptime tokenize.parseSimple("42").?.match;
    try std.testing.expectEqual(1, tokens.len);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[0].type);
    try std.testing.expectEqualStrings("42", tokens[0].lexeme);
}

test "tokenize: decimal number" {
    const tokens = comptime tokenize.parseSimple("3.14").?.match;
    try std.testing.expectEqual(1, tokens.len);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[0].type);
    try std.testing.expectEqualStrings("3.14", tokens[0].lexeme);
}

test "tokenize: addition" {
    const tokens = comptime tokenize.parseSimple("1+2").?.match;
    try std.testing.expectEqual(3, tokens.len);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[0].type);
    try std.testing.expectEqualStrings("1", tokens[0].lexeme);
    try std.testing.expectEqual(TokenType.PLUS, tokens[1].type);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[2].type);
    try std.testing.expectEqualStrings("2", tokens[2].lexeme);
}

test "tokenize: multiplication with spaces" {
    const tokens = comptime tokenize.parseSimple("2 * 3").?.match;
    try std.testing.expectEqual(3, tokens.len);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[0].type);
    try std.testing.expectEqual(TokenType.STAR, tokens[1].type);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[2].type);
}

test "tokenize: parentheses" {
    const tokens = comptime tokenize.parseSimple("(1+2)").?.match;
    try std.testing.expectEqual(5, tokens.len);
    try std.testing.expectEqual(TokenType.LPAREN, tokens[0].type);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[1].type);
    try std.testing.expectEqual(TokenType.PLUS, tokens[2].type);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[3].type);
    try std.testing.expectEqual(TokenType.RPAREN, tokens[4].type);
}

test "tokenize: invalid character" {
    const tokens = comptime tokenize.parseSimple("1@2").?.match;
    try std.testing.expectEqual(3, tokens.len);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[0].type);
    try std.testing.expectEqual(TokenType.INVALID, tokens[1].type);
    try std.testing.expectEqualStrings("@", tokens[1].lexeme);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[2].type);
}

test "token parser: number token" {
    const tokens = comptime tokenize.parseSimple("42 +").?.match;
    try std.testing.expectEqual(2, tokens.len);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[0].type);
    try std.testing.expectEqualStrings("42", tokens[0].lexeme);
    try std.testing.expectEqual(TokenType.PLUS, tokens[1].type);
}

test "token parser: plus token" {
    const tokens = comptime tokenize.parseSimple("+ 42").?.match;
    try std.testing.expectEqual(2, tokens.len);
    try std.testing.expectEqual(TokenType.PLUS, tokens[0].type);
    try std.testing.expectEqualStrings("+", tokens[0].lexeme);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[1].type);
}

test "token parser: delimited example" {
    const tokens = comptime tokenize.parseSimple("(42)").?.match;
    try std.testing.expectEqual(3, tokens.len);
    try std.testing.expectEqual(TokenType.LPAREN, tokens[0].type);
    try std.testing.expectEqual(TokenType.NUMBER, tokens[1].type);
    try std.testing.expectEqualStrings("42", tokens[1].lexeme);
    try std.testing.expectEqual(TokenType.RPAREN, tokens[2].type);
}

test "parse: single number" {
    const input = "42";
    const tokens = comptime tokenize.parseSimple(input).?.match;
    const result = comptime parseExpression(tokens, input);
    try std.testing.expect(result.len == 1);
    try std.testing.expect(result[0].kind != .Error);
    try std.testing.expect(result[0].kind == .Scalar);
    try std.testing.expectEqual(42.0, result[0].kind.Scalar);
}

test "parse: addition" {
    const input = "1+2";
    const tokens = comptime tokenize.parseSimple(input).?.match;
    const result = comptime parseExpression(tokens, input);
    try std.testing.expect(result.len == 3); // root + 2 children
    try std.testing.expect(result[0].kind == .BinOp);
    try std.testing.expectEqual(OpType.ADD, result[0].kind.BinOp);
    try std.testing.expect(result[1].kind == .Scalar);
    try std.testing.expectEqual(1.0, result[1].kind.Scalar);
    try std.testing.expect(result[2].kind == .Scalar);
    try std.testing.expectEqual(2.0, result[2].kind.Scalar);
}

test "parse: subtraction" {
    const input = "5-3";
    const tokens = comptime tokenize.parseSimple(input).?.match;
    const result = comptime parseExpression(tokens, input);
    try std.testing.expect(result.len == 3); // root + 2 children
    try std.testing.expect(result[0].kind == .BinOp);
    try std.testing.expectEqual(OpType.SUB, result[0].kind.BinOp);
    try std.testing.expect(result[1].kind == .Scalar);
    try std.testing.expectEqual(5.0, result[1].kind.Scalar);
    try std.testing.expect(result[2].kind == .Scalar);
    try std.testing.expectEqual(3.0, result[2].kind.Scalar);
}

test "parse: multiplication" {
    const input = "2*3";
    const tokens = comptime tokenize.parseSimple(input).?.match;
    const result = comptime parseExpression(tokens, input);
    try std.testing.expect(result.len == 3);
    try std.testing.expect(result[0].kind == .BinOp);
    try std.testing.expectEqual(OpType.MUL, result[0].kind.BinOp);
    try std.testing.expectEqual(2.0, result[1].kind.Scalar);
    try std.testing.expectEqual(3.0, result[2].kind.Scalar);
}

// TODO: Update these tests for array-based AST
test "parse: precedence (multiplication before addition)" {
    const input = "1+2*3";
    const tokens = comptime tokenize.parseSimple(input).?.match;
    const result = comptime parseExpression(tokens, input);

    // Should parse as: 1 + (2 * 3)
    // Array layout: [ADD, 1, MUL, 2, 3]
    try std.testing.expect(result.len == 5);

    // Root is ADD
    try std.testing.expect(result[0].kind == .BinOp);
    try std.testing.expectEqual(OpType.ADD, result[0].kind.BinOp);

    // Left child of ADD is 1
    try std.testing.expect(result[1].kind == .Scalar);
    try std.testing.expectEqual(1.0, result[1].kind.Scalar);

    // Right child of ADD is MUL subtree starting at index 2
    try std.testing.expect(result[2].kind == .BinOp);
    try std.testing.expectEqual(OpType.MUL, result[2].kind.BinOp);

    // Left child of MUL is 2
    try std.testing.expect(result[3].kind == .Scalar);
    try std.testing.expectEqual(2.0, result[3].kind.Scalar);

    // Right child of MUL is 3
    try std.testing.expect(result[4].kind == .Scalar);
    try std.testing.expectEqual(3.0, result[4].kind.Scalar);
}

test "parse: precedence (multiple multiplications)" {
    const input = "2*3+4";
    const tokens = comptime tokenize.parseSimple(input).?.match;
    const result = comptime parseExpression(tokens, input);

    // Should parse as: (2 * 3) + 4
    // Array layout: [ADD, MUL, 2, 3, 4]
    try std.testing.expect(result.len == 5);

    // Root is ADD
    try std.testing.expect(result[0].kind == .BinOp);
    try std.testing.expectEqual(OpType.ADD, result[0].kind.BinOp);

    // Left child of ADD is MUL subtree starting at index 1
    try std.testing.expect(result[1].kind == .BinOp);
    try std.testing.expectEqual(OpType.MUL, result[1].kind.BinOp);

    // Left child of MUL is 2
    try std.testing.expect(result[2].kind == .Scalar);
    try std.testing.expectEqual(2.0, result[2].kind.Scalar);

    // Right child of MUL is 3
    try std.testing.expect(result[3].kind == .Scalar);
    try std.testing.expectEqual(3.0, result[3].kind.Scalar);

    // Right child of ADD is 4
    try std.testing.expect(result[4].kind == .Scalar);
    try std.testing.expectEqual(4.0, result[4].kind.Scalar);
}

test "parse: parentheses override precedence" {
    const input = "(1+2)*3";
    const tokens = comptime tokenize.parseSimple(input).?.match;
    const result = comptime parseExpression(tokens, input);

    // Should parse as: (1 + 2) * 3
    // Array layout: [MUL, ADD, 1, 2, 3]
    try std.testing.expect(result.len == 5);

    // Root is MUL
    try std.testing.expect(result[0].kind == .BinOp);
    try std.testing.expectEqual(OpType.MUL, result[0].kind.BinOp);

    // Left child of MUL is ADD subtree starting at index 1
    try std.testing.expect(result[1].kind == .BinOp);
    try std.testing.expectEqual(OpType.ADD, result[1].kind.BinOp);

    // Left child of ADD is 1
    try std.testing.expect(result[2].kind == .Scalar);
    try std.testing.expectEqual(1.0, result[2].kind.Scalar);

    // Right child of ADD is 2
    try std.testing.expect(result[3].kind == .Scalar);
    try std.testing.expectEqual(2.0, result[3].kind.Scalar);

    // Right child of MUL is 3
    try std.testing.expect(result[4].kind == .Scalar);
    try std.testing.expectEqual(3.0, result[4].kind.Scalar);
}

test "parse: nested parentheses" {
    const input = "((1+2)*(3+4))";
    const tokens = comptime tokenize.parseSimple(input).?.match;
    const result = comptime parseExpression(tokens, input);

    // Should parse as: (1 + 2) * (3 + 4)
    // Array layout: [MUL, ADD, 1, 2, ADD, 3, 4]
    try std.testing.expect(result.len == 7);

    // Root is MUL
    try std.testing.expect(result[0].kind == .BinOp);
    try std.testing.expectEqual(OpType.MUL, result[0].kind.BinOp);

    // Left child of MUL is first ADD subtree starting at index 1
    try std.testing.expect(result[1].kind == .BinOp);
    try std.testing.expectEqual(OpType.ADD, result[1].kind.BinOp);

    // Children of first ADD
    try std.testing.expect(result[2].kind == .Scalar);
    try std.testing.expectEqual(1.0, result[2].kind.Scalar);
    try std.testing.expect(result[3].kind == .Scalar);
    try std.testing.expectEqual(2.0, result[3].kind.Scalar);

    // Right child of MUL is second ADD subtree starting at index 4
    try std.testing.expect(result[4].kind == .BinOp);
    try std.testing.expectEqual(OpType.ADD, result[4].kind.BinOp);

    // Children of second ADD
    try std.testing.expect(result[5].kind == .Scalar);
    try std.testing.expectEqual(3.0, result[5].kind.Scalar);
    try std.testing.expect(result[6].kind == .Scalar);
    try std.testing.expectEqual(4.0, result[6].kind.Scalar);
}

test "parse: error on invalid token" {
    const input = "1@2";
    const tokens = comptime tokenize.parseSimple(input).?.match;
    // The tokenizer will mark @ as INVALID, but the parser might still try to parse it
    const result = comptime parseExpression(tokens, input);
    // Since @ is tokenized as INVALID, the parser should fail when it encounters it
    // The error should be in the result array
    try std.testing.expect(result.len > 0);
    // Check if we have an error somewhere in the tree
    var has_error = false;
    for (result) |node| {
        if (node.kind == .Error) {
            has_error = true;
            break;
        }
    }
    try std.testing.expect(has_error);
}

test "parse: missing closing parenthesis" {
    const input = "(1+2";
    const tokens = comptime tokenize.parseSimple(input).?.match;
    const result = comptime parseExpression(tokens, input);
    // Should return an error node
    try std.testing.expect(result.len == 1);
    try std.testing.expect(result[0].kind == .Error);
    try std.testing.expectEqual(ParseErrorKind.MissingClosingParen, result[0].kind.Error.kind);
}

test "parse: unexpected closing parenthesis" {
    const input = "1+2)";
    const tokens = comptime tokenize.parseSimple(input).?.match;
    const result = comptime parseExpression(tokens, input);
    // This should give us an error about unexpected tokens after expression
    try std.testing.expect(result.len == 1);
    try std.testing.expect(result[0].kind == .Error);
    try std.testing.expectEqual(ParseErrorKind.InvalidExpression, result[0].kind.Error.kind);
}

// Test the new combinator-based parser
test "combinator parse: single number" {
    const input = "42";
    const tokens = comptime tokenize.parseSimple(input).?.match;
    const result = comptime parseExpressionCombinator(tokens, input);
    try std.testing.expect(result.len == 1);
    try std.testing.expect(result[0].kind != .Error);
    try std.testing.expect(result[0].kind == .Scalar);
    try std.testing.expectEqual(42.0, result[0].kind.Scalar);
}

test "combinator parse: addition" {
    const input = "1+2";
    const tokens = comptime tokenize.parseSimple(input).?.match;
    const result = comptime parseExpressionCombinator(tokens, input);
    try std.testing.expect(result.len == 3); // root + 2 children
    try std.testing.expect(result[0].kind == .BinOp);
    try std.testing.expectEqual(OpType.ADD, result[0].kind.BinOp);
    try std.testing.expect(result[1].kind == .Scalar);
    try std.testing.expectEqual(1.0, result[1].kind.Scalar);
    try std.testing.expect(result[2].kind == .Scalar);
    try std.testing.expectEqual(2.0, result[2].kind.Scalar);
}

test "combinator parse: multiplication precedence" {
    const input = "1+2*3";
    const tokens = comptime tokenize.parseSimple(input).?.match;
    const result = comptime parseExpressionCombinator(tokens, input);

    // Should parse as: 1 + (2 * 3)
    // Array layout: [ADD, 1, MUL, 2, 3]
    try std.testing.expect(result.len == 5);

    // Root is ADD
    try std.testing.expect(result[0].kind == .BinOp);
    try std.testing.expectEqual(OpType.ADD, result[0].kind.BinOp);

    // Left child of ADD is 1
    try std.testing.expect(result[1].kind == .Scalar);
    try std.testing.expectEqual(1.0, result[1].kind.Scalar);

    // Right child of ADD is MUL subtree starting at index 2
    try std.testing.expect(result[2].kind == .BinOp);
    try std.testing.expectEqual(OpType.MUL, result[2].kind.BinOp);

    // Left child of MUL is 2
    try std.testing.expect(result[3].kind == .Scalar);
    try std.testing.expectEqual(2.0, result[3].kind.Scalar);

    // Right child of MUL is 3
    try std.testing.expect(result[4].kind == .Scalar);
    try std.testing.expectEqual(3.0, result[4].kind.Scalar);
}

test "combinator parse: parentheses override precedence" {
    const input = "(1+2)*3";
    const tokens = comptime tokenize.parseSimple(input).?.match;
    const result = comptime parseExpressionCombinator(tokens, input);

    // Should parse as: (1 + 2) * 3
    // Array layout: [MUL, ADD, 1, 2, 3]
    try std.testing.expect(result.len == 5);

    // Root is MUL
    try std.testing.expect(result[0].kind == .BinOp);
    try std.testing.expectEqual(OpType.MUL, result[0].kind.BinOp);

    // Left child of MUL is ADD subtree starting at index 1
    try std.testing.expect(result[1].kind == .BinOp);
    try std.testing.expectEqual(OpType.ADD, result[1].kind.BinOp);

    // Left child of ADD is 1
    try std.testing.expect(result[2].kind == .Scalar);
    try std.testing.expectEqual(1.0, result[2].kind.Scalar);

    // Right child of ADD is 2
    try std.testing.expect(result[3].kind == .Scalar);
    try std.testing.expectEqual(2.0, result[3].kind.Scalar);

    // Right child of MUL is 3
    try std.testing.expect(result[4].kind == .Scalar);
    try std.testing.expectEqual(3.0, result[4].kind.Scalar);
}
