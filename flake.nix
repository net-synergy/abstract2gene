{
  description = "Word distributions related to gene symbols";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, poetry2nix }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ poetry2nix.overlays.default ];
        config.cudaSupport = true;
        config.allowUnfreePredicate = pkg:
          builtins.elem (pkgs.lib.getName pkg) [
            "cuda-merged"
            "cuda_cuobjdump"
            "cudnn"
            "cuda_gdb"
            "cuda_nvcc"
            "cuda_nvdisasm"
            "cuda_nvprune"
            "cuda_cccl"
            "cuda_cudart"
            "cuda_cupti"
            "cuda_cuxxfilt"
            "cuda_nvml_dev"
            "cuda_nvrtc"
            "cuda_nvtx"
            "cuda_profiler_api"
            "cuda_sanitizer_api"
            "libcublas"
            "libcufft"
            "libcurand"
            "libcusolver"
            "libnvjitlink"
            "libcusparse"
            "libnpp"
            "nvidia-settings"
            "nvidia-x11"
          ];
      };

      synstore = (pkgs.python3Packages.buildPythonPackage rec {
        pname = "synstore";
        version = "devel";
        format = "pyproject";

        nativeBuildInputs = [ pkgs.python3Packages.poetry-core ];
        buildInputs = [ pkgs.python3Packages.platformdirs ];
        src = pkgs.fetchFromGitHub {
          owner = "net-synergy";
          repo = "synstore";
          rev = "devel";
          sha256 = "sha256-uOzkVvVb+dkmqnLEYUeUlfaBlJbNobdKITTViT/Ecvc=";
        };
        # src = pkgs.fetchPypi {
        #   inherit pname version;
        #   sha256 = "sha256-8O+8a9VzCloxWJGMkoDHDnKy4aCP6srjDkHcPf45eM8=";
        # };
      });

      # a2gEnv = pkgs.poetry2nix.mkPoetryEnv {
      #   projectDir = ./.;
      #   editablePackageSources = { abstract2gene = ./.; };
      #   preferWheels = true;
      # overrides = pkgs.poetry2nix.defaultPoetryOverrides.extend
      #   (final: prev: {
      #     jax = prev.jax.overridePythonAttrs (old: { cudaSupport = true; });
      #     jaxlib =
      #       prev.jaxlib.overridePythonAttrs (old: { cudaSupport = true; });
      #   });
      # };

      a2gEnv = (pkgs.python3.withPackages (ps:
        with ps; [
          ipython
          black
          isort
          python-lsp-server
          pydocstyle
          pylsp-mypy
          python-lsp-ruff
          ipdb

          pandas
          scikit-learn

          pytorch
          jax
          flax
          optax
          numpy
          tqdm
          transformers
          datasets
          peft
          scipy
          pyarrow

          synstore
        ]));

      REnv = (pkgs.rWrapper.override {
        packages = with pkgs.rPackages; [
          styler
          lintr

          tidyverse
        ];
      });
    in {
      devShells.${system}.default =
        pkgs.mkShell { packages = [ a2gEnv pkgs.poetry REnv ]; };
    };
}
