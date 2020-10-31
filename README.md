# Prerequisites

1. [Docker](https://www.docker.com)

# Preparations

1. Download `assets.zip` from latest [releases](https://github.com/miggymigz/mimic-predictors-fastapi/releases). `assets.zip` contains 3 directories: `saved_models`, `js_scripts`, and `mimic-predictors-ui`.

2. Extract the archive and place the 3 directories as shown below:

   - saved_models → `services/tf-serving/saved_models`
   - js_scripts → `services/mongodb/js_scripts`
   - mimic-predictors-ui → `services/app-client/mimic-predictors-ui`

3. Proceed to build/running the containers.

# How to Build/Run

1. Execute the following command to build/run the containers.

   ```bash
   docker-compose up --build
   ```
   
2. Go to [http://localhost](http://localhost)
