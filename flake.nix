{
  description = "Development environment for cliff";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
          name = "zig/cliff";

          buildInputs = with pkgs; [
            gh
            lldb
            man-pages
            man-pages-posix
            zig
            zls
          ];

          nativeBuildInputs = with pkgs; [
            clang-tools
          ];

        };
      }
    );
}
