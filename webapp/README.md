# Web App

## Running the webapp

Docker compose can be used to start up the webapp.
Two example docker compose files and an environment file are located under the `docker` subdirectory.
One docker is CPU only and the other takes advantage of available GPUs.
To use, you can copy one of the docker compose files and the `.env` file to the project root and modify as needed.
This will require a downloading the full dataset and pre-embedding all the abstracts.
It will also require that the abstract2gene model has already been trained (and you may want to set the model used in the TOML above).
As such the first run will take a long time to initialize but should start up much faster on subsequent runs.
See the `webapp/README.md` for more information.

Assuming the repository has already been cloned and the current working directory is the repository.

``` shell
docker compose up --build
```

## Configuration

There are two places for configuring the webapp: the `a2g.toml` and the docker `.env` file.

### a2g.toml

To modify the behavior of the UI, a `a2g.toml` can be created at the root of the project.

An example with all values set:

``` toml
[engine]
labels_per_batch = 16

[ui]
min_genes_displayed = 5
gene_thresh = 0.5
results_per_page = 20

[auth]
enabled = true
```

### env

For the most part the `.env` file should probably be left alone with exception to changing the port. The `A2G_PORT` can be set to change which port (on the host) the docker is listening on. To open the webapp go to `localhost:$PORT` where port is the value in `A2G_PORT`.

## Users

Authentication is enabled by default. To disable, set `auth.enabled` to `false` (notice it most be lowercase unlike in python) in `a2g.toml`.
When enabled, initially there are no register users so it will not be possible to log in.
Users can be defined by creating a `users.json` under the `./auth` (in the docker this should be mounted to `/root/.local/share/auth`).
Users should be a dictionary of usernames to hashed passwords.
The `example/usergen.py` script can be used to help with user management.
It will create the `auth/users.json` if it doesn't exist.
The script accepts the commands `list`, `add`, and `rm` to list the existing usernames, add a new user, and remove an existing user.

## Installation

## First run

The first run will take a long time to start since the abstract data needs to be downloaded and the qdrant DB will be created by running the model on all the abstracts.

Once this has been completed, the qdrant DB will be saved so future starts will be quick.
