variable "image_tag" {
  type = string
  description = "The docker image tag of the current deployment."
}

job "quantsystem" {
  group "dashboard" {
    count = 1

    network {
      port "dashboard_port" {
        to = 8050
      }
    }
    
    task "dashboard" {
      driver = "docker"

      config {
        # The IMAGE_TAG environment variable will be interpolated at runtime.
        image = "ghcr.io/mackhaymond/quantsystem:${var.image_tag}"
        ports = ["dashboard_port"]
      }

      service {
        name = "wwm"
        port = "dashboard_port"
        tags = [
          "traefik.enable=true",
        ]

        check {
          name     = "dashboard-check"
          type     = "http"
          path     = "/"
          interval = "10s"
          timeout  = "2s"
        }
      }
    }
  }
}
