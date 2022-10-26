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
          format = "pyproject";
          buildInputs = (with python.pkgs; [ poetry ]);
          propagatedBuildInputs = [ pubnet.packages.${system}.pubnet ]
            ++ (with python.pkgs; [ pandas nltk ]);
          authors = [ "David Connell <davidconnell12@gmail.com>" ];
          checkPhase = "";
        };
        nix2poetryDependency = list:
          builtins.concatStringsSep "\n" (builtins.map (dep:
            let
              pname = if dep.pname == "python3" then "python" else dep.pname;
              versionList = builtins.splitVersion dep.version;
              major = builtins.elemAt versionList 0;
              minor = builtins.elemAt versionList 1;
              version = if pname == "python" then
                ''\"~${major}.${minor}\"''
              else
                ''\"^${major}.${minor}\"'';
            in pname + " = " + version) list);
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

            if [ ! -f pyproject.toml ] || \
               [ $(date +%s -r flake.nix) -gt $(date +%s -r pyproject.toml) ]; then
               pname=${abstract2gene.pname} \
               version=${abstract2gene.version} \
               description='Word distributions related to gene symbols' \
               license=MIT \
               authors="${
                 builtins.concatStringsSep ",\n    "
                 (builtins.map (name: ''\"'' + name + ''\"'')
                   abstract2gene.authors)
               }" \
               dependencies="${
                 nix2poetryDependency abstract2gene.propagatedBuildInputs
               }" ./.pyproject.toml.template
            fi
          '';
        };
      });
}
