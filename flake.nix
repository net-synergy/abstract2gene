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

            speakeasy2 = (prev.buildPythonPackage rec {
              pname = "speakeasy2";
              version = "0.1.4";
              format = "wheel";

              dist = "cp312";
              python = "cp312";
              abi = "manylinux_2_35";
              platform = "x86_64";

              src = pkgs.fetchurl {
                url =
                  "https://files.pythonhosted.org/packages/f3/bd/12e8504531b9b2049535e566a7ddf497f01fe236bc1fe58249509bae7313/${pname}-${version}-${dist}-${python}-${abi}_${platform}.whl";
                sha256 = "sha256-NAK3pKHCqBWDKY6PQPdQmxOWrIHfDL8h3GQWZEKUzkI=";
              };

              propagatedBuildInputs = [ prev.igraph prev.numpy ];
            });

            pubmedparser2 = (prev.buildPythonPackage rec {
              pname = "pubmedparser2";
              version = "2.1.2";
              format = "wheel";

              dist = "cp312";
              python = "cp312";
              abi = "manylinux_2_35";
              platform = "x86_64";

              src = pkgs.fetchurl {
                url =
                  "https://files.pythonhosted.org/packages/43/9c/bed728b6ba9c31ff622bbc151c5a9d6ed3c0097a1c8f954292dc04a36078/${pname}-${version}-${dist}-${python}-${abi}_${platform}.whl";
                sha256 =
                  "6ed156ad7e607c183d46c2f25acf23cb153d8d48ef0a5ffe061c041408bacb8e";
              };

              propagatedBuildInputs = [ prev.appdirs ];
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
        groups = [ "dev" "app" ];
      });

      tex = pkgs.texlive.combine {
        inherit (pkgs.texlive)
          scheme-small latex-bin latexmk tex-gyre tex-gyre-math type1cm cm-super
          microtype fpl palatino mathpazo;
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ a2gEnv tex ];

        env = {
          UV_NO_SYNC = "1";
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON = python.interpreter;
          PYTHONPATH = ".";
        };
      };
    };
}
