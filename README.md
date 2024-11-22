# QuantSystem

## Docker Instructions

### Build the Docker Image

To build the Docker image, run the following command in the root of the repository:

```sh
docker build -t quantsystem .
```

### Run the Docker Container

To run the Docker container, use the following command:

```sh
docker run -p 8050:8050 -e PORT=8050 quantsystem
```

This will start the Dash app and make it accessible at `http://localhost:8050`.
