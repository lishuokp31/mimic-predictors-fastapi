# Prerequisites

1. [Docker](https://www.docker.com)

# Preparations

1. Download `assets.zip` from latest [releases](https://github.com/miggymigz/mimic-predictors-fastapi/releases). `assets.zip` contains 3 directories: `saved_models`, `js_scripts`, and `mimic-predictors-ui`.

2. Extract the archive and place the 3 directories as shown below:

   - saved_models → `services/tf-serving/saved_models`
   - js_scripts → `services/mongodb/js_scripts`
   - mimic-predictors-ui → `services/app-client/mimic-predictors-ui`

3. Proceed to build/running the containers.

# How to Build/Run (using docker-compose)

1. Add expose directives to the `tf-serving` and `mongodb` services (in `docker-compose.yml`).

   ```yml
   # mongodb
   expose:
     - 27017
   ---
   # tf-serving
   expose:
     - 5000
   ```

1. Attach Traefik labels' to containers (as opposed to attaching them to services) (in `docker-compose.prod.yml`).

   ```yml
   services:
     frontend:
       labels:
         - "traefik.enable=true"
   ```

1. Remove Traefik swarm mode (in `docker-compose.traefik.yml`).

   ```yml
   # delete this line in the command group
   command:
     - "--providers.docker.swarmmode=true"
   ```

1. Remove port information labels for the frontend and backend (in `docker-compose.prod.yml`).

   ```yml
   services:
     frontend:
       deploy:
         labels:
           - "traefik.http.services.frontend.loadbalancer.server.port=80"

     backend:
       deploy:
         labels:
           - "traefik.http.services.backend.loadbalancer.server.port=80"
   ```

1. Remove manager node constraint (in `docker-compose.traefik.yml`).

   ```yml
   deploy:
     placement:
       constraints:
         - node.role == manager
   ```

1. Execute the following command to build/run the containers.

   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.traefik.yml -f docker-compose.prod.yml up -d
   ```

1. Go to [http://localhost/app/](http://localhost/app/)

# How to Deploy (using docker swarm; default mode)

1. Execute the following command to build/run the containers.

   ```bash
   docker stack deploy -c docker-compose.yml -c docker-compose.traefik.yml -c docker-compose.prod.yml mimic-predictors
   ```

1. Go to [http://localhost/app/](http://localhost/app/)
