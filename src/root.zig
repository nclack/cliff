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
        basis: u8,
        /// the magnitude of this blade.
        magnitude: Field,

        /// Compare two blades for approximate equality
        pub fn approxEql(self: Self, other: Self, tolerance: Field) bool {
            if (self.basis != other.basis) return false;
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
            if (self.basis != 0) {
                try writer.writeAll(" e");
                inline for (0..8) |i| {
                    if ((self.basis >> @intCast(i)) & 1 == 1) {
                        try writer.print("{d}", .{i});
                    }
                }
            }
        }

        pub fn mul(self: Self, rhs: anytype) Self {
            const RhsType = @TypeOf(rhs);
            return switch (RhsType) {
                Field, comptime_float, comptime_int => .{
                    .basis = self.basis,
                    .magnitude = self.magnitude * rhs,
                },
                Self => blk: {
                    const metric = SelfMetric{};
                    const squares: u8 = self.basis & rhs.basis;
                    if ((squares & metric.zero) != 0) {
                        break :blk SelfBasis.zero;
                    }

                    const swap: f32 = if ((swaps(self.basis, rhs.basis, SelfMetric.rank) & 1) != 0) -1 else 1;
                    const sign: f32 = if ((@popCount(squares & metric.negative) & 1) != 0) -1 else 1;

                    break :blk .{
                        .basis = self.basis ^ rhs.basis,
                        .magnitude = self.magnitude * rhs.magnitude * swap * sign,
                    };
                },
                *Self, *const Self => self.mul(rhs.*),
                else => @compileError("mul expects either a Blade or a scalar (Field type)"),
            };
        }

        /// Add (returns Multivector; supports Blade, Multivector, scalar, and pointers)
        pub fn add(self: Self, rhs: anytype) Multivector(sig, Field) {
            const RhsType = @TypeOf(rhs);
            const MV = Multivector(sig, Field);
            return switch (RhsType) {
                Field, comptime_float, comptime_int => blk: {
                    var result = MV.fromBlade(self);
                    result.coefficients[0] += rhs;
                    break :blk result;
                },
                Self => blk: {
                    var result = MV.fromBlade(self);
                    result.coefficients[rhs.basis] += rhs.magnitude;
                    break :blk result;
                },
                MV => blk: {
                    const mv = MV.fromBlade(self);
                    break :blk mv.add(rhs);
                },
                *Self, *const Self => self.add(rhs.*),
                *MV, *const MV => self.add(rhs.*),
                else => @compileError("add expects a Blade, Multivector, or scalar (Field type)"),
            };
        }
    };
}

/// A Multivector is a sum of blades represented densely with SIMD coefficients.
/// For a rank-n algebra, stores 2^n coefficients indexed by blade signature.
pub fn Multivector(comptime sig: Signature, comptime Field: type) type {
    const rank = sig.rank();
    const size = 1 << rank;

    return struct {
        const Self = @This();
        const SelfBlade = Blade(sig, Field);
        const SelfMetric = Metric(sig);

        /// Dense array of coefficients indexed by signature.
        /// coefficients[i] = coefficient for blade with signature i
        coefficients: @Vector(size, Field),

        /// Initialize a zero multivector
        pub fn zero() Self {
            return .{ .coefficients = @splat(0.0) };
        }

        /// Create a multivector from a single blade
        pub fn fromBlade(blade: SelfBlade) Self {
            var result = zero();
            result.coefficients[blade.basis] = blade.magnitude;
            return result;
        }

        /// Add (supports Multivector, Blade, scalar, and pointers)
        pub fn add(self: Self, rhs: anytype) Self {
            const RhsType = @TypeOf(rhs);
            return switch (RhsType) {
                Field, comptime_float, comptime_int => blk: {
                    var result = self;
                    result.coefficients[0] += rhs;
                    break :blk result;
                },
                SelfBlade => blk: {
                    const mv = Self.fromBlade(rhs);
                    break :blk self.add(mv);
                },
                Self => .{ .coefficients = self.coefficients + rhs.coefficients },
                *Self, *const Self => self.add(rhs.*),
                *SelfBlade, *const SelfBlade => self.add(rhs.*),
                else => @compileError("add expects a Multivector, Blade, or scalar (Field type)"),
            };
        }

        /// Compare two multivectors for approximate equality
        pub fn approxEql(self: Self, other: Self, tolerance: Field) bool {
            const diff = @abs(self.coefficients - other.coefficients);
            const tol_vec: @Vector(size, Field) = @splat(tolerance);
            const within_tolerance = diff <= tol_vec;
            return @reduce(.And, within_tolerance);
        }

        /// Select only blades of a specific grade
        pub fn selectGrade(self: Self, comptime grade: u4) Self {
            const mask = comptime blk: {
                var m: [size]Field = undefined;
                for (0..size) |i| {
                    const i_grade = @popCount(i);
                    m[i] = if (i_grade == grade) 1.0 else 0.0;
                }
                break :blk m;
            };
            return .{ .coefficients = self.coefficients * mask };
        }

        /// Geometric product (supports Multivector, Blade, scalar, and pointers)
        pub fn mul(self: Self, rhs: anytype) Self {
            const RhsType = @TypeOf(rhs);
            return switch (RhsType) {
                Field, comptime_float, comptime_int => blk: {
                    const splat: @Vector(size, Field) = @splat(rhs);
                    break :blk .{ .coefficients = self.coefficients * splat };
                },
                SelfBlade => blk: {
                    const mv = Self.fromBlade(rhs);
                    break :blk self.mul(mv);
                },
                Self => blk: {
                    var result = zero();
                    const metric = SelfMetric{};

                    // Direct computation: multiply each pair of basis blades
                    for (0..size) |i| {
                        const sig_i: u8 = @intCast(i);
                        const a_coef = self.coefficients[i];
                        if (a_coef == 0.0) continue;

                        for (0..size) |j| {
                            const sig_j: u8 = @intCast(j);
                            const b_coef = rhs.coefficients[j];
                            if (b_coef == 0.0) continue;

                            // Check if product is zero (squared degenerate vector)
                            const squares: u8 = sig_i & sig_j;
                            if ((squares & metric.zero) != 0) {
                                continue;
                            }

                            // Compute sign from swaps and negative squares
                            const swap_sign: Field = if ((swaps(sig_i, sig_j, rank) & 1) != 0) -1.0 else 1.0;
                            const metric_sign: Field = if ((@popCount(squares & metric.negative) & 1) != 0) -1.0 else 1.0;

                            // Result signature and magnitude
                            const result_sig = sig_i ^ sig_j;
                            const magnitude = a_coef * b_coef * swap_sign * metric_sign;

                            result.coefficients[result_sig] += magnitude;
                        }
                    }

                    break :blk result;
                },
                *Self, *const Self => self.mul(rhs.*),
                *SelfBlade, *const SelfBlade => self.mul(rhs.*),
                else => @compileError("mul expects a Multivector, Blade, or scalar (Field type)"),
            };
        }
    };
}

/// Convert basis bits to field name (e.g., 0b0101 -> "e20")
/// Iterates from high bit to low bit for intuitive swap counting
fn signatureToName(comptime signature: u8, comptime rank: u8) [:0]const u8 {
    if (signature == 0) return "scalar";

    comptime var name: []const u8 = "e";
    comptime var i: u8 = rank;
    inline while (i > 0) {
        i -= 1;
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
        defs[0] = .{ .basis = 0, .magnitude = 0.0 };

        // pseudoscalar
        defs[1] = .{ .basis = 0xff, .magnitude = 1.0 };

        // All 2^rank blades
        // The first is one: a scalar with magnitude 1
        for (0..num_blades) |basis_bits| {
            defs[basis_bits + 2] = .{ .basis = @intCast(basis_bits), .magnitude = 1.0 };
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

/// Test helper: check multivector equality with diagnostic output on failure
fn expectMultivectorEqual(result: anytype, expected: @TypeOf(result), tolerance: f32) !void {
    if (!result.approxEql(expected, tolerance)) {
        std.debug.print("\n  Expected coefficients: {any}\n", .{expected.coefficients});
        std.debug.print("  Got coefficients:      {any}\n", .{result.coefficients});
        return error.TestExpectedEqual;
    }
}

test "Euclidean: e0 * e0 = 1" {
    var a = euclidean.e0;
    const result = a.mul(a);
    try expectBladeEqual(result, euclidean.one, 1e-6);
}

test "Euclidean: e0 * e1 = -e10" {
    var a = euclidean.e0;
    const b = euclidean.e1;
    const result = a.mul(b);
    const expected = euclidean.e10.mul(-1.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Euclidean: e1 * e0 = e10" {
    var a = euclidean.e1;
    const b = euclidean.e0;
    const result = a.mul(b);
    try expectBladeEqual(result, euclidean.e10, 1e-6);
}

test "Euclidean: e1 * e320 = e3210" {
    var a = euclidean.e1;
    const b = euclidean.e320;
    const result = a.mul(b);
    try expectBladeEqual(result, euclidean.e3210, 1e-6);
}

test "Euclidean: e420 * e1 = -e4210" {
    var a = euclidean.e420;
    const b = euclidean.e1;
    const result = a.mul(b);
    try expectBladeEqual(result, euclidean.e4210.mul(-1), 1e-6);
}

test "Euclidean: e1 * e20 = -e210" {
    var a = euclidean.e1;
    const b = euclidean.e20;
    const result = a.mul(b);
    const expected = euclidean.e210.mul(-1.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Euclidean: coefficient propagation 2.5*e0 * 3.0*e1 = -7.5*e10" {
    var a = euclidean.e0.mul(2.5);
    const b = euclidean.e1.mul(3.0);
    const result = a.mul(b);
    const expected = euclidean.e10.mul(-7.5);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Euclidean: negative coefficient -2.0*e0 * 3.0*e1 = 6.0*e10" {
    var a = euclidean.e0.mul(-2.0);
    const b = euclidean.e1.mul(3.0);
    const result = a.mul(b);
    const expected = euclidean.e10.mul(6.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Euclidean: pseudoscalar squared = 1" {
    var a = euclidean.pseudoscalar;
    const result = a.mul(a);
    try expectBladeEqual(result, euclidean.one, 1e-6);
}

test "Euclidean: e10 squared = -1" {
    var a = euclidean.e10;
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

test "Minkowski: e3 * e30 = -e0" {
    var a = minkowski3d.e3;
    const b = minkowski3d.e30;
    const result = a.mul(b);
    const expected = minkowski3d.e0.mul(-1.0);
    try expectBladeEqual(result, expected, 1e-6);
}

test "Conformal: e4 * e4 = 0 (degenerate)" {
    var a = conformal3d.e4.mul(5.0);
    const result = a.mul(a);
    try expectBladeEqual(result, conformal3d.zero, 1e-6);
}

test "Conformal: e40 * e40 = 0 (degenerate)" {
    var a = conformal3d.e40.mul(3.0);
    const result = a.mul(a);
    try expectBladeEqual(result, conformal3d.zero, 1e-6);
}

// Multivector tests
const Euclidean3dMV = Multivector(.{ .positive = 3, .negative = 0, .zero = 0 }, f32);

test "Multivector: addition" {
    const a = Euclidean3dMV.fromBlade(euclidean3d.e0.mul(2.0));
    const b = Euclidean3dMV.fromBlade(euclidean3d.e1.mul(3.0));
    const result = a.add(b);
    const expected = Euclidean3dMV{ .coefficients = .{ 0, 2, 3, 0, 0, 0, 0, 0 } };
    try expectMultivectorEqual(result, expected, 1e-6);
}

test "Multivector: scalar multiplication" {
    const a = Euclidean3dMV.fromBlade(euclidean3d.e0.mul(2.0));
    const result = a.mul(3.0);
    const expected = Euclidean3dMV{ .coefficients = .{ 0, 6, 0, 0, 0, 0, 0, 0 } };
    try expectMultivectorEqual(result, expected, 1e-6);
}

test "Multivector: grade select scalar" {
    const mv = Euclidean3dMV{ .coefficients = .{ 5, 2, 3, 1, 0, 0, 0, 0 } };
    const result = mv.selectGrade(0);
    const expected = Euclidean3dMV{ .coefficients = .{ 5, 0, 0, 0, 0, 0, 0, 0 } };
    try expectMultivectorEqual(result, expected, 1e-6);
}

test "Multivector: grade select vectors" {
    const mv = Euclidean3dMV{ .coefficients = .{ 5, 2, 3, 1, 0, 0, 0, 0 } };
    const result = mv.selectGrade(1);
    const expected = Euclidean3dMV{ .coefficients = .{ 0, 2, 3, 0, 0, 0, 0, 0 } };
    try expectMultivectorEqual(result, expected, 1e-6);
}

test "Multivector: geometric product matches blade product" {
    const a = Euclidean3dMV{ .coefficients = .{ 0, 1, 0, 0, 0, 0, 0, 0 } }; // e0
    const b = Euclidean3dMV{ .coefficients = .{ 0, 0, 1, 0, 0, 0, 0, 0 } }; // e1
    const result = a.mul(b);
    // e0 * e1 = -e10 (basis 3)
    const expected = Euclidean3dMV{ .coefficients = .{ 0, 0, 0, -1, 0, 0, 0, 0 } };
    try expectMultivectorEqual(result, expected, 1e-6);
}

test "Multivector: (e0 + e1) * e2" {
    const a = Euclidean3dMV{ .coefficients = .{ 0, 1, 1, 0, 0, 0, 0, 0 } }; // e0 + e1
    const result = a.mul(euclidean3d.e2);
    // e0 * e2 = -e20 (basis 5), e1 * e2 = -e21 (basis 6)
    const expected = Euclidean3dMV{ .coefficients = .{ 0, 0, 0, 0, 0, -1, -1, 0 } };
    try expectMultivectorEqual(result, expected, 1e-6);
}

test "Multivector: (2*e0 + 3*e1) * (4*e0 + 5*e1)" {
    const a = Euclidean3dMV{ .coefficients = .{ 0, 2, 3, 0, 0, 0, 0, 0 } };
    const b = Euclidean3dMV{ .coefficients = .{ 0, 4, 5, 0, 0, 0, 0, 0 } };
    const result = a.mul(b);
    // e0*e0 = 1, e1*e1 = 1 => scalar: 2*4 + 3*5 = 23
    // e0*e1 = -e10, e1*e0 = e10 => e10: -2*5 + 3*4 = 2
    const expected = Euclidean3dMV{ .coefficients = .{ 23, 0, 0, 2, 0, 0, 0, 0 } };
    try expectMultivectorEqual(result, expected, 1e-6);
}

// Generic operation tests
test "Multivector: add with scalar" {
    const a = Euclidean3dMV{ .coefficients = .{ 0, 2, 0, 0, 0, 0, 0, 0 } };
    const result = a.add(5.0);
    const expected = Euclidean3dMV{ .coefficients = .{ 5, 2, 0, 0, 0, 0, 0, 0 } };
    try expectMultivectorEqual(result, expected, 1e-6);
}
