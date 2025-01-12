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

      pubmedparser2 = (pkgs.python3Packages.buildPythonPackage rec {
        pname = "pubmedparser2";
        version = "2.1.2";
        format = "wheel";

        src = pkgs.fetchPypi rec {
          inherit pname version format;
          sha256 = "sha256-btFWrX5gfBg9RsLyWs8jyxU9jUjvCl/+BhwEFAi6y44=";
          dist = python;
          python = "cp312";
          abi = "cp312";
          platform = "manylinux_2_35_x86_64";
        };
      });

      pubnet = (pkgs.python3Packages.buildPythonPackage rec {
        pname = "pubnet";
        version = "0.9.1";
        format = "pyproject";

        nativeBuildInputs = [ pkgs.python3Packages.poetry-core ];
        propagatedBuildInputs = (with pkgs.python3Packages; [
          appdirs
          igraph
          matplotlib
          numpy
          pandas
          pubmedparser2
          scipy
        ]);
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-wjPKH+qIC8Mf/rRFsffQ+R0QyY8Qs5z5fd6X2Mkzsaw=";
        };
      });

      synstore = (pkgs.python3Packages.buildPythonPackage rec {
        pname = "synstore";
        version = "0.1.2";
        format = "pyproject";

        nativeBuildInputs = [ pkgs.python3Packages.poetry-core ];
        buildInputs = [ pkgs.python3Packages.platformdirs ];
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-8O+8a9VzCloxWJGMkoDHDnKy4aCP6srjDkHcPf45eM8=";
        };
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

          jax
          jaxlib
          flax
          optax
          numpy
          tqdm
          transformers
          peft
          scipy

          synstore
          pubnet
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
