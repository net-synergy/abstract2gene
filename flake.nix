{
  description = "Word distributions related to gene symbols";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-22.05";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pubnet = {
      url = "gitlab:DavidRConnell/pubnet";
      # url = "/home/voidee/packages/python/pubnet";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, pubnet }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;
        abstract2gene = python.pkgs.buildPythonPackage rec {
          pname = "abstract2gene";
          version = "0.1.0";
          src = ./.;
          propagatedBuildInputs = [ pubnet.packages.${system}.pubnet ]
            ++ (with python.pkgs; [ pandas nltk pytest ]);
          preBuild = ''
            cat >setup.py <<_EOF_
            from setuptools import setup
            setup(
                name='${pname}',
                version='${version}',
                license='MIT',
                description="Word distributions related to gene symbols",
                packages={'${pname}'},
                install_requires=[
                'pandas',
                'nltk',
                'pubnet',
                ],
                tests_require=['pytest']
            )
            _EOF_
          '';
        };
      in {
        packages.abstract2gene = abstract2gene;
        packages.default = self.packages.${system}.abstract2gene;
        devShells.default = pkgs.mkShell {
          packages = [
            (python.withPackages (p:
              with p;
              [
                # development dependencies
                ipython
                python-lsp-server
                pyls-isort
                python-lsp-black
                pylsp-mypy
              ] ++ abstract2gene.propagatedBuildInputs))
          ];
          shellHook = ''
            export PYTHONPATH=.
          '';
        };
      });
}
