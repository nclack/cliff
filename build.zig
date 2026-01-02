const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const opt = b.standardOptimizeOption(.{});

    const lib = b.addLibrary(.{
        .name = "cliff",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = opt,
        }),
    });
    b.installArtifact(lib);

    const run_tests = b.step("test", "Run tests");

    // Unit tests
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
        }),
    });
    run_tests.dependOn(&b.addRunArtifact(unit_tests).step);

    // Integration tests
    const integration_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/integration.zig"),
            .target = target,
        }),
    });
    integration_tests.root_module.addImport("cliff", lib.root_module);
    run_tests.dependOn(&b.addRunArtifact(integration_tests).step);
}
