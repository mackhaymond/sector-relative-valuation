packer {
  required_plugins {
    docker = {
      version = ">= 1.0.8"
      source  = "github.com/hashicorp/docker"
    }
  }
}

variable "registry" {
  type    = string
}

variable "image_name" {
  type = string
}

variable "image_tag" {
  type = string
}

variable "latest_tag" {
  type    = string
}

source "docker" "quantsystem" {
  image  = "python:3.12-slim"
  commit = true
  changes = [
    "WORKDIR /app",
    "EXPOSE 8050",
    "CMD [\"python\", \"src/dashboard.py\"]"
  ]
}

build {
  name = "quantsystem-build"
  sources = ["source.docker.quantsystem"]

  provisioner "file" {
    source      = "."
    destination = "/app"
  }

  provisioner "shell" {
    inline = [
      "apt-get update",
      "apt-get install -y --no-install-recommends gcc g++ make",
      "pip install poetry",
      "cd /app",
      "poetry config virtualenvs.create false",
      "poetry install --no-dev",
      "apt-get remove -y gcc g++ make",
      "apt-get autoremove -y",
      "apt-get clean",
      "rm -rf /var/lib/apt/lists/*"
    ]
  }

  post-processors {
    post-processor "docker-tag" {
      repository = "${var.registry}/${var.image_name}"
      tags       = [var.image_tag, var.latest_tag]
    }

    post-processor "docker-push" {}
  }
}
