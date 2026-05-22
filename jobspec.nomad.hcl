variable "image_tag" {
  type        = string
  description = "The docker image tag of the current deployment."
}

job "sector-relative-valuation" {
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
        image = "ghcr.io/mackhaymond/sector-relative-valuation:${var.image_tag}"
        ports = ["dashboard_port"]
      }

      service {
        name = "sector-relative-valuation"
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
