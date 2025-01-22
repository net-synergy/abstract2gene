{
  description = "Classify abstracts with gene annotations";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, pyproject-nix }:
    let
      system = "x86_64-linux";

      overlay = final: prev: {
        python3 = prev.python3.override {
          packageOverrides = final: prev: {
            plotnine = (prev.buildPythonPackage rec {
              pname = "plotnine";
              version = "0.14.5";
              format = "pyproject";

              src = prev.fetchPypi {
                inherit pname version;
                sha256 = "sha256-nnWWno4Q2NdwpL420Q4HXMELiMpvzJnjatpTQ2+1ZT8=";
              };

              nativeBuildInputs = [ prev.setuptools_scm ];
              propagatedBuildInputs = with prev; [
                matplotlib
                pandas
                mizani
                numpy
                scipy
                statsmodels
              ];
            });

            synstore = (prev.buildPythonPackage rec {
              pname = "synstore";
              version = "0.1.3";
              format = "pyproject";

              src = prev.fetchPypi {
                inherit pname version;
                sha256 = "sha256-PED+Z+BlptuoQVOGvG8EICxohX+2stbXyJdwNqap32I=";
              };

              nativeBuildInputs = [ prev.poetry-core ];
              propagatedBuildInputs = [ prev.platformdirs ];
            });
          };
        };
      };

      pkgs = import nixpkgs {
        inherit system;
        overlays = [ overlay ];
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

      project = pyproject-nix.lib.project.loadPyproject { projectRoot = ./.; };
      python = pkgs.python3;

      a2gEnv = python.withPackages (project.renderers.withPackages {
        inherit python;
        groups = [ "dev" ];
      });

      REnv = (pkgs.rWrapper.override {
        packages = with pkgs.rPackages; [
          styler
          lintr

          tidyverse
        ];
      });
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ REnv a2gEnv ];

        env = {
          UV_NO_SYNC = "1";
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON = python.interpreter;
          PYTHONPATH = ".";
        };
      };
    };
}
