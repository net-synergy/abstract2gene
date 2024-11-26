{
  description = "Word distributions related to gene symbols";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    vim = {
      url = "gitlab:DavidRConnell/vim-container";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, poetry2nix, vim }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ poetry2nix.overlays.default ];
      };

      a2gEnv = pkgs.poetry2nix.mkPoetryEnv {
        projectDir = ./.;
        editablePackageSources = { abstract2gene = ./.; };
        preferWheels = true;
      };

      REnv = (pkgs.rWrapper.override {
        packages = with pkgs.rPackages; [
          styler
          lintr

          tidyverse
        ];
      });

      nixDockerImage = pkgs.dockerTools.buildImage {
        name = "abstract2gene";
        tag = "nix";
        fromImage = vim.packages.${system}.dockerImage;
        config = {
          Env = [
            "HOME=/home/docker"
            "XDG_DATA_HOME=/home/docker/.local/share"
            "XDG_CACHE_HOME=/home/docker/.cache"
            "XDG_CONFIG_HOME=/home/docker/.config"
          ];
        };
        contents = [
          a2gEnv
          (pkgs.buildEnv {
            name = "image-root";
            pathsToLink = [ "/bin" ];
            paths = [ pkgs.bashInteractive pkgs.coreutils pkgs.tmux ];
          })
        ];
      };
    in {
      packages.dockerImage = nixDockerImage;
      devShells.${system}.default =
        pkgs.mkShell { packages = [ a2gEnv pkgs.poetry REnv ]; };
    };
}
