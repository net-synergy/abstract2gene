{
  description = "Word distributions related to gene symbols";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-22.11";
    flake-utils = { url = "github:numtide/flake-utils"; };
    vim = {
      url = "gitlab:DavidRConnell/vim-container";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, vim }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        a2gEnv = pkgs.poetry2nix.mkPoetryEnv {
          projectDir = ./.;
          editablePackageSources = { abstract2gene = ./abstract2gene; };
          preferWheels = true;
          extraPackages = (ps:
            with ps; [
              ipython
              python-lsp-server
              pyls-isort
              python-lsp-black
              pylsp-mypy
            ]);
          groups = [ ];
        };
        abstract2gene = pkgs.poetry2nix.mkPoetryPackage { projectDir = ./.; };
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
        packages.abstract2gene = abstract2gene;
        packages.default = self.packages.${system}.abstract2gene;
        packages.dockerImage = nixDockerImage;
        devShells.default = pkgs.mkShell { packages = [ a2gEnv pkgs.poetry ]; };
      });
}
