# To learn more about how to use Nix to configure your environment
# see: https://firebase.google.com/docs/studio/customize-workspace
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-24.05"; # or "unstable"

  # Use https://search.nixos.org/packages to find packages
  packages = [
    pkgs.python312
    pkgs.rustup
    pkgs.pkg-config
    # Required for cpal (audio) on Linux
    pkgs.alsa-lib
    # Required for pyqt6
    pkgs.libxkbcommon
  ];

  # Sets environment variables in the workspace
  env = {};

  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      "ms-python.python"
      "rust-lang.rust-analyzer"
    ];

    # Enable previews
    previews = {
      enable = true;
    };

    # Workspace lifecycle hooks
    workspace = {
      # Runs when a workspace is first created
      onCreate = {
        install-pip-deps = "pip install -r requirements.txt";
        install-rust-toolchain = "rustup default stable";
        build-rust-engine = "cd rust_engine && maturin develop";
      };
      # Runs when the workspace is (re)started
      onStart = {
        # Ensure Rust engine is built
        build-rust = "cd rust_engine && maturin develop";
      };
    };
  };
}
