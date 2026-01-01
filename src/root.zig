const std = @import("std");

/// O(#bits) algorithm for counting the swaps required for the permutation
/// ordering the basis vectors of a and b.
///
/// That is if a = e_k..e_0 and b = e_j..e_0 then ab = e_k..e_0 e_j .. e_0.
/// We want to know the swaps required to order those so we can rewrite
/// ab = e_k .. e_j .. e_0 e_0 (assuming k>=j in this example).
fn swaps(a: u8, b: u8) u8 {
    var acc: u8 = 0;
    inline for (1..8) |i| {
        acc += @popCount(a & (b >> i));
    }
    return acc;
}

const MetricSignature = struct {
    positive: u8,
    zero: u8,
    negative: u8,
};

/// Make the metric
pub fn Metric(comptime sig: MetricSignature) type {
    // Use high bits for vectors squaring to 0
    // Use low bits for vectors squaring to 1
    // Use middle bits for vectors squaring to -1
    //
    // The metric just stores the masks for the vectors squaring to 0 or -1.
    return struct {
        const Self = @This();

        comptime negative: u8 = ((1 << sig.negative) - 1) << sig.positive,
        comptime zero: u8 = ((1 << sig.zero) - 1) << (sig.positive + sig.negative),
    };
}

/// A Blade is a scalar magnitude with a product of ordered unit vectors.
pub fn Blade(comptime sig: MetricSignature, comptime Field: type) type {
    return struct {
        const Self = @This();
        const SelfMetric = Metric(sig);
        const SelfBasis = Basis(sig, Field);

        /// bit i is true if the blade has basis vector i
        signature: u8,
        /// the magnitude of this blade.
        magnitude: Field,

        /// Scale this blade by a scalar
        pub fn scale(self: Self, s: Field) Self {
            return .{
                .signature = self.signature,
                .magnitude = self.magnitude * s,
            };
        }

        /// Compare two blades for approximate equality
        pub fn approxEql(self: Self, other: Self, tolerance: Field) bool {
            if (self.signature != other.signature) return false;
            const diff = @abs(self.magnitude - other.magnitude);
            return diff <= tolerance;
        }

        /// Format a blade for printing (e.g., "2.5*e014")
        pub fn format(
            self: Self,
            writer: *std.io.Writer,
        ) !void {
            // Print magnitude
            try writer.print("{d}", .{self.magnitude});

            // Print basis vectors in concatenated form
            if (self.signature != 0) {
                try writer.writeAll(" e");
                inline for (0..8) |i| {
                    if ((self.signature >> @intCast(i)) & 1 == 1) {
                        try writer.print("{d}", .{i});
                    }
                }
            }
        }

        pub fn mul(lhs: *Self, rhs: *Self) Self {
            const metric = SelfMetric{};
            const squares: u8 = lhs.signature & rhs.signature;
            if ((squares & metric.zero) != 0) {
                return SelfBasis.zero;
            }

            const swap: f32 = if ((swaps(lhs.signature, rhs.signature) & 1) != 0) -1 else 1;
            const sign: f32 = if ((@popCount(squares & metric.negative) & 1) != 0) -1 else 1;

            return .{
                .signature = lhs.signature ^ rhs.signature,
                .magnitude = lhs.magnitude * rhs.magnitude * swap * sign,
            };
        }
    };
}

/// Unit basis blades with magnitude 1.0
pub fn Basis(comptime sig: MetricSignature, comptime Field: type) type {
    return struct {
        pub const zero: Blade(sig, Field) = .{ .signature = 0x0, .magnitude = 0.0 };
        pub const one: Blade(sig, Field) = .{ .signature = 0x0, .magnitude = 1.0 };

        pub const pseudoscalar: Blade(sig, Field) = .{ .signature = 0xff, .magnitude = 1.0 };

        // Single basis vectors
        pub const e0: Blade(sig, Field) = .{ .signature = 0b00000001, .magnitude = 1.0 };
        pub const e1: Blade(sig, Field) = .{ .signature = 0b00000010, .magnitude = 1.0 };
        pub const e2: Blade(sig, Field) = .{ .signature = 0b00000100, .magnitude = 1.0 };
        pub const e3: Blade(sig, Field) = .{ .signature = 0b00001000, .magnitude = 1.0 };
        pub const e4: Blade(sig, Field) = .{ .signature = 0b00010000, .magnitude = 1.0 };
        pub const e5: Blade(sig, Field) = .{ .signature = 0b00100000, .magnitude = 1.0 };
        pub const e6: Blade(sig, Field) = .{ .signature = 0b01000000, .magnitude = 1.0 };
        pub const e7: Blade(sig, Field) = .{ .signature = 0b10000000, .magnitude = 1.0 };

        // Multi-index blades for tests
        pub const e01: Blade(sig, Field) = .{ .signature = 0b00000011, .magnitude = 1.0 };
        pub const e02: Blade(sig, Field) = .{ .signature = 0b00000101, .magnitude = 1.0 };
        pub const e03: Blade(sig, Field) = .{ .signature = 0b00001001, .magnitude = 1.0 };
        pub const e04: Blade(sig, Field) = .{ .signature = 0b00010001, .magnitude = 1.0 };
        pub const e012: Blade(sig, Field) = .{ .signature = 0b00000111, .magnitude = 1.0 };
        pub const e023: Blade(sig, Field) = .{ .signature = 0b00001101, .magnitude = 1.0 };
        pub const e0123: Blade(sig, Field) = .{ .signature = 0b00001111, .magnitude = 1.0 };
    };
}

pub const euclidean = Basis(.{ .positive = 8, .zero = 0, .negative = 0 }, f32);
pub const minkowski3d = Basis(.{ .positive = 3, .zero = 0, .negative = 1 }, f32);
pub const conformal3d = Basis(.{ .positive = 3, .zero = 1, .negative = 1 }, f32);

/// Test helper: check blade equality with diagnostic output on failure
fn expectBladeEqual(result: anytype, expected: @TypeOf(result), tolerance: f32) !void {
    if (!result.approxEql(expected, tolerance)) {
        std.debug.print("\n  Expected: {f}\n", .{expected});
        std.debug.print("  Got:      {f}\n", .{result});
        return error.TestExpectedEqual;
    }
}

test "Euclidian: scalar multiplication" {
    var a = euclidean.one.scale(2.0);
    var b = euclidean.e0.scale(3.0);
    const result = a.mul(&b);
    const expected = euclidean.e0.scale(6.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Euclidean: e0 * e0 = 1" {
    var a = euclidean.e0;
    const result = a.mul(&a);
    try expectBladeEqual(result, euclidean.one, 1e-6);
}

test "Euclidean: e0 * e1 = -e01" {
    var a = euclidean.e0;
    var b = euclidean.e1;
    const result = a.mul(&b);
    const expected = euclidean.e01.scale(-1.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Euclidean: e1 * e0 = e01" {
    var a = euclidean.e1;
    var b = euclidean.e0;
    const result = a.mul(&b);
    try expectBladeEqual(result, euclidean.e01, 1e-6);
}

test "Euclidean: e1 * e023 = e0123" {
    var a = euclidean.e1;
    var b = euclidean.e023;
    const result = a.mul(&b);
    try expectBladeEqual(result, euclidean.e0123, 1e-6);
}

test "Euclidean: e1 * e02 = -e012" {
    var a = euclidean.e1;
    var b = euclidean.e02;
    const result = a.mul(&b);
    const expected = euclidean.e012.scale(-1.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Euclidean: coefficient propagation 2.5*e0 * 3.0*e1 = -7.5*e01" {
    var a = euclidean.e0.scale(2.5);
    var b = euclidean.e1.scale(3.0);
    const result = a.mul(&b);
    const expected = euclidean.e01.scale(-7.5);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Euclidean: negative coefficient -2.0*e0 * 3.0*e1 = 6.0*e01" {
    var a = euclidean.e0.scale(-2.0);
    var b = euclidean.e1.scale(3.0);
    const result = a.mul(&b);
    const expected = euclidean.e01.scale(6.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Euclidean: pseudoscalar squared = 1" {
    var a = euclidean.pseudoscalar;
    const result = a.mul(&a);
    try expectBladeEqual(result, euclidean.one, 1e-6);
}

test "Euclidean: e01 squared = -1" {
    var a = euclidean.e01;
    const result = a.mul(&a);
    const expected = euclidean.one.scale(-1.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Minkowski: e3 * e3 = -1 (timelike)" {
    var a = minkowski3d.e3;
    const result = a.mul(&a);
    const expected = minkowski3d.one.scale(-1.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Minkowski: e0 * e0 = 1 (spacelike)" {
    var a = minkowski3d.e0;
    const result = a.mul(&a);
    try expectBladeEqual(result, minkowski3d.one, 1e-6);
}

test "Minkowski: e3 * e03 = -e0" {
    var a = minkowski3d.e3;
    var b = minkowski3d.e03;
    const result = a.mul(&b);
    const expected = minkowski3d.e0.scale(-1.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Conformal: e4 * e4 = 0 (degenerate)" {
    var a = conformal3d.e4.scale(5.0);
    const result = a.mul(&a);
    try expectBladeEqual(result, conformal3d.zero, 1e-6);
}

test "Conformal: e04 * e04 = 0 (degenerate)" {
    var a = conformal3d.e04.scale(3.0);
    const result = a.mul(&a);
    try expectBladeEqual(result, conformal3d.zero, 1e-6);
}
