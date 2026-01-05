const std = @import("std");
const cliff = @import("cliff");
const dsl = @import("dsl");

test "Multivector api access" {
    _ = cliff.euclidean3d.Multivector.zero();
}

// DSL tests

// const e3d = cliff.Cl(3, 0, 0);

// test "DSL: parse scalar literal" {
//     const result = dsl.eval(e3d, "3.5", .{});
//     try std.testing.expectEqual(3.5, result.coefficients[0]);
//     try std.testing.expectEqual(0.0, result.coefficients[1]);
// }

// test "DSL: parse basis blade" {
//     const result = dsl.eval(e3d, "e0", .{});
//     try std.testing.expectEqual(0.0, result.coefficients[0]);
//     try std.testing.expectEqual(1.0, result.coefficients[1]);
// }

// test "DSL: scalar addition" {
//     const result = dsl.eval(e3d, "1+2", .{});
//     try std.testing.expectEqual(3.0, result.coefficients[0]);
// }

// test "DSL: scalar multiplication" {
//     const result = dsl.eval(e3d, "2*3", .{});
//     try std.testing.expectEqual(6.0, result.coefficients[0]);
// }

// test "DSL: basis blade multiplication" {
//     const result = dsl.eval(e3d, "e0*e1", .{});
//     // e0 * e1 = -e10 (basis 3)
//     try std.testing.expectEqual(0.0, result.coefficients[0]);
//     try std.testing.expectEqual(-1.0, result.coefficients[3]);
// }

// test "DSL: with scalar variable" {
//     const result = dsl.eval(e3d, "a+1", .{ .a = 2.0 });
//     try std.testing.expectEqual(3.0, result.coefficients[0]);
// }

// test "DSL: with blade variable" {
//     const result = dsl.eval(e3d, "b*e0", .{ .b = cliff.euclidean3d.e1 });
//     // e1 * e0 = e10 (basis 3)
//     try std.testing.expectEqual(1.0, result.coefficients[3]);
// }

// test "DSL: complex expression" {
//     const result = dsl.eval(e3d, "a+e0*(e21+b*e210)", .{ .a = 1.0, .b = 3.0 });
//     // Should evaluate: 1 + e0*(e21 + 3*e210)
//     //                = 1 + e0*e21 + 3*e0*e210
//     //                = 1 + e021 + 3*e0210
//     //                = 1 - e210 + 3*e210   (e021 = -e210, e0210 = e210)
//     //                = 1 + 2*e210
//     try std.testing.expectEqual(1.0, result.coefficients[0]);
//     try std.testing.expectEqual(2.0, result.coefficients[7]); // e210
// }

// test "DSL: parentheses" {
//     const result = dsl.eval(e3d, "(1+2)*3", .{});
//     try std.testing.expectEqual(9.0, result.coefficients[0]);
// }
