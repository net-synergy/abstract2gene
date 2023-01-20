{
  description = "Word distributions related to gene symbols";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-22.11";
    flake-utils = { url = "github:numtide/flake-utils"; };
    pubnet = {
      url = "gitlab:DavidRConnell/pubnet";
      # url = "/home/voidee/packages/python/pubnet";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
    vim = {
      url = "gitlab:DavidRConnell/vim-container";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, pubnet, vim }:
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
            ++ (with python.pkgs; [ pandas nltk requests ]);
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
            (python.withPackages (p:
              abstract2gene.propagatedBuildInputs ++ (with p; [
                ipython
                python-lsp-server
                pyls-isort
                python-lsp-black
                pylsp-mypy
              ])))
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

            if [ ! -f devShell.sh ]; then
               echo "#!/usr/bin/env bash" > devShell.sh
               echo "" >> devShell.sh
               echo "cd \$(dirname \$0)" >> devShell.sh
               echo "" >> devShell.sh
               echo "docker run --rm -v \$(pwd):/home/docker/package -v \$XDG_DATA_HOME:/home/docker/.local/share -v \$XDG_CACHE_HOME:/home/docker/.cache -w /home/docker/package -it abstract2gene:nix ipython" >> devShell.sh
               chmod +x devShell.sh
            fi
          '';
        };
      });
}
