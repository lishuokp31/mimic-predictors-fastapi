# Prerequisites

1. [Docker](https://www.docker.com)

# How to Build

```bash
docker-compose build
```

# How to Run

Run all containers in a detached mode (Terminal will just invoke their startup script and exit):

```bash
docker-compose up -d
```

Run all containers and see its progress:

```bash
docker-compose up
```

Build and run all containers (remove `-d` if you don't want to detach):

```bash
docker-compose up --build -d
```
