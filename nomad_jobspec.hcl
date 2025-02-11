job "quantsystem" {
  group "dashboard" {
    count = 3

    network {
      port "dashboard_port" {
        to = 8050
      }
    }
    
    task "dashboard" {
      driver = "docker"

      config {
        # The IMAGE_TAG environment variable will be interpolated at runtime.
        image = "ghcr.io/spyicydev/quantsystem:${IMAGE_TAG}"
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
