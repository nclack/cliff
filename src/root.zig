const std = @import("std");

/// O(#bits) algorithm for counting the swaps required for the permutation
/// ordering the basis vectors of a and b.
///
/// That is if a = e_k..e_0 and b = e_j..e_0 then ab = e_k..e_0 e_j .. e_0.
/// We want to know the swaps required to order those so we can rewrite
/// ab = e_k .. e_j .. e_0 e_0 (assuming k>=j in this example).
fn swaps(a: u8, b: u8, comptime rank: u8) u8 {
    var acc: u8 = 0;
    inline for (1..rank) |i| {
        acc += @popCount(a & (b >> i));
    }
    return acc;
}

const Signature = struct {
    const Self = @This();

    positive: u8,
    negative: u8,
    zero: u8,

    fn rank(comptime self: Self) u8 {
        return self.positive + self.negative + self.zero;
    }
};

/// Make the metric
pub fn Metric(comptime sig: Signature) type {
    // Use high bits for vectors squaring to 0
    // Use low bits for vectors squaring to 1
    // Use middle bits for vectors squaring to -1
    //
    // The metric just stores the masks for the vectors squaring to 0 or -1.
    comptime {
        if (sig.rank() > 8) {
            @compileError("Algebras must have at most 8 dimensions.");
        }
    }
    return struct {
        const Self = @This();

        const rank: u8 = sig.rank();
        comptime negative: u8 = ((1 << sig.negative) - 1) << sig.positive,
        comptime zero: u8 = ((1 << sig.zero) - 1) << (sig.positive + sig.negative),
    };
}

/// A Blade is a scalar magnitude with a product of ordered unit vectors.
pub fn Blade(comptime sig: Signature, comptime Field: type) type {
    return struct {
        const Self = @This();
        const SelfMetric = Metric(sig);
        const SelfBasis = Basis(sig, Field);

        /// bit i is true if the blade has basis vector i
        signature: u8,
        /// the magnitude of this blade.
        magnitude: Field,

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

        pub fn mul(self: Self, rhs: anytype) Self {
            const RhsType = @TypeOf(rhs);
            return switch (RhsType) {
                Field, comptime_float, comptime_int => .{
                    .signature = self.signature,
                    .magnitude = self.magnitude * rhs,
                },
                Self => blk: {
                    const metric = SelfMetric{};
                    const squares: u8 = self.signature & rhs.signature;
                    if ((squares & metric.zero) != 0) {
                        break :blk SelfBasis.zero;
                    }

                    const swap: f32 = if ((swaps(self.signature, rhs.signature, SelfMetric.rank) & 1) != 0) -1 else 1;
                    const sign: f32 = if ((@popCount(squares & metric.negative) & 1) != 0) -1 else 1;

                    break :blk .{
                        .signature = self.signature ^ rhs.signature,
                        .magnitude = self.magnitude * rhs.magnitude * swap * sign,
                    };
                },
                *Self, *const Self => self.mul(rhs.*),
                else => @compileError("mul expects either a Blade or a scalar (Field type)"),
            };
        }
    };
}

/// Convert signature bits to field name (e.g., 0b0101 -> "e02")
fn signatureToName(comptime signature: u8, comptime rank: u8) [:0]const u8 {
    if (signature == 0) return "scalar";

    comptime var name: []const u8 = "e";
    inline for (0..rank) |i| {
        if ((signature >> @as(u4, @intCast(i))) & 1 == 1) {
            const digit: []const u8 = switch (i) {
                0 => "0",
                1 => "1",
                2 => "2",
                3 => "3",
                4 => "4",
                5 => "5",
                6 => "6",
                7 => "7",
                else => unreachable,
            };
            name = name ++ digit;
        }
    }
    // Ensure null termination
    const final: [:0]const u8 = name ++ "";
    return final;
}

/// Unit basis blades with magnitude 1.0
/// Returns an instance with all 2^rank basis blades auto-generated
pub fn Basis(comptime sig: Signature, comptime Field: type) BasisInstance(sig, Field) {
    return .{};
}

/// Generate a type with the 2^rank unit blades
pub fn BasisInstance(comptime sig: Signature, comptime Field: type) type {
    @setEvalBranchQuota(10000);
    const rank = sig.rank();
    const num_blades = @as(usize, 1) << @as(u4, @intCast(rank));

    // Create persistent array of default values
    const defaults = comptime blk: {
        var defs: [num_blades + 2]Blade(sig, Field) = undefined;

        // zero: scalar with magnitude 0
        defs[0] = .{ .signature = 0, .magnitude = 0.0 };

        // pseudoscalar
        defs[1] = .{ .signature = 0xff, .magnitude = 1.0 };

        // All 2^rank blades
        // The first is one: a scalar with magnitude 1
        for (0..num_blades) |signature| {
            defs[signature + 2] = .{ .signature = @intCast(signature), .magnitude = 1.0 };
        }

        break :blk defs;
    };

    // Build struct fields for all possible blades
    var fields: [num_blades + 2]std.builtin.Type.StructField = undefined;

    // Special fields
    fields[0] = .{
        .name = "zero",
        .type = Blade(sig, Field),
        .default_value_ptr = @ptrCast(&defaults[0]),
        .is_comptime = true,
        .alignment = @alignOf(Blade(sig, Field)),
    };

    fields[1] = .{
        .name = "pseudoscalar",
        .type = Blade(sig, Field),
        .default_value_ptr = @ptrCast(&defaults[1]),
        .is_comptime = true,
        .alignment = @alignOf(Blade(sig, Field)),
    };

    fields[2] = .{
        .name = "one",
        .type = Blade(sig, Field),
        .default_value_ptr = @ptrCast(&defaults[2]),
        .is_comptime = true,
        .alignment = @alignOf(Blade(sig, Field)),
    };

    // Generate all 2^rank blades
    inline for (1..num_blades) |signature| {
        const sig_u8: u8 = @intCast(signature);
        const name = signatureToName(sig_u8, rank);

        fields[signature + 2] = .{
            .name = name,
            .type = Blade(sig, Field),
            .default_value_ptr = @ptrCast(&defaults[signature + 2]),
            .is_comptime = true,
            .alignment = @alignOf(Blade(sig, Field)),
        };
    }

    return @Type(.{
        .@"struct" = .{
            .layout = .auto,
            .fields = &fields,
            .decls = &.{},
            .is_tuple = false,
        },
    });
}

pub const euclidean = Basis(.{ .positive = 8, .negative = 0, .zero = 0 }, f32);
pub const euclidean3d = Basis(.{ .positive = 3, .negative = 0, .zero = 0 }, f32);
pub const minkowski3d = Basis(.{ .positive = 3, .negative = 1, .zero = 0 }, f32);
pub const projective2d = Basis(.{ .positive = 2, .negative = 0, .zero = 1 }, f32);
pub const conformal2d = Basis(.{ .positive = 2, .negative = 1, .zero = 1 }, f32);
pub const projective3d = Basis(.{ .positive = 3, .negative = 0, .zero = 1 }, f32);
pub const conformal3d = Basis(.{ .positive = 3, .negative = 1, .zero = 1 }, f32);

/// Test helper: check blade equality with diagnostic output on failure
fn expectBladeEqual(result: anytype, expected: @TypeOf(result), tolerance: f32) !void {
    if (!result.approxEql(expected, tolerance)) {
        std.debug.print("\n  Expected: {f}\n", .{expected});
        std.debug.print("  Got:      {f}\n", .{result});
        return error.TestExpectedEqual;
    }
}

test "Euclidian: scalar multiplication" {
    var a = euclidean.one.mul(2.0);
    const b = euclidean.e0.mul(3.0);
    const result = a.mul(b);
    const expected = euclidean.e0.mul(6.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Euclidean: e0 * e0 = 1" {
    var a = euclidean.e0;
    const result = a.mul(a);
    try expectBladeEqual(result, euclidean.one, 1e-6);
}

test "Euclidean: e0 * e1 = -e01" {
    var a = euclidean.e0;
    const b = euclidean.e1;
    const result = a.mul(b);
    const expected = euclidean.e01.mul(-1.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Euclidean: e1 * e0 = e01" {
    var a = euclidean.e1;
    const b = euclidean.e0;
    const result = a.mul(b);
    try expectBladeEqual(result, euclidean.e01, 1e-6);
}

test "Euclidean: e1 * e023 = e0123" {
    var a = euclidean.e1;
    const b = euclidean.e023;
    const result = a.mul(b);
    try expectBladeEqual(result, euclidean.e0123, 1e-6);
}

test "Euclidean: e1 * e02 = -e012" {
    var a = euclidean.e1;
    const b = euclidean.e02;
    const result = a.mul(b);
    const expected = euclidean.e012.mul(-1.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Euclidean: coefficient propagation 2.5*e0 * 3.0*e1 = -7.5*e01" {
    var a = euclidean.e0.mul(2.5);
    const b = euclidean.e1.mul(3.0);
    const result = a.mul(b);
    const expected = euclidean.e01.mul(-7.5);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Euclidean: negative coefficient -2.0*e0 * 3.0*e1 = 6.0*e01" {
    var a = euclidean.e0.mul(-2.0);
    const b = euclidean.e1.mul(3.0);
    const result = a.mul(b);
    const expected = euclidean.e01.mul(6.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Euclidean: pseudoscalar squared = 1" {
    var a = euclidean.pseudoscalar;
    const result = a.mul(a);
    try expectBladeEqual(result, euclidean.one, 1e-6);
}

test "Euclidean: e01 squared = -1" {
    var a = euclidean.e01;
    const result = a.mul(a);
    const expected = euclidean.one.mul(-1.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Minkowski: e3 * e3 = -1 (timelike)" {
    var a = minkowski3d.e3;
    const result = a.mul(a);
    const expected = minkowski3d.one.mul(-1.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Minkowski: e0 * e0 = 1 (spacelike)" {
    var a = minkowski3d.e0;
    const result = a.mul(a);
    try expectBladeEqual(result, minkowski3d.one, 1e-6);
}

test "Minkowski: e3 * e03 = -e0" {
    var a = minkowski3d.e3;
    const b = minkowski3d.e03;
    const result = a.mul(b);
    const expected = minkowski3d.e0.mul(-1.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Conformal: e4 * e4 = 0 (degenerate)" {
    var a = conformal3d.e4.mul(5.0);
    const result = a.mul(a);
    try expectBladeEqual(result, conformal3d.zero, 1e-6);
}

test "Conformal: e04 * e04 = 0 (degenerate)" {
    var a = conformal3d.e04.mul(3.0);
    const result = a.mul(a);
    try expectBladeEqual(result, conformal3d.zero, 1e-6);
}
